import contextlib
import importlib
import pathlib
import shutil
import sys
import traceback
from typing import Optional

import pandas as pd
import pyomo.environ as pyo
import typer
from loguru import logger
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition

from new_modeling_toolkit import __version__
from new_modeling_toolkit.core import stream
from new_modeling_toolkit.core.utils.core_utils import timer
from new_modeling_toolkit.core.utils.util import DirStructure
from new_modeling_toolkit.resolve import export_results
from new_modeling_toolkit.resolve import model_formulation
from new_modeling_toolkit.resolve.export_results_summary import export_all_results_summary
from new_modeling_toolkit.resolve.model_formulation import ResolveCase


@timer
def solve(
    resolve_model: model_formulation.ResolveCase,
    dir_str: DirStructure,
    solver_name: str,
    log_level: str,
    symbolic_solver_labels: bool,
):
    """Initialize specified solver, associated solver options & solve model."""

    if solver_name == "gurobi":
        solver = SolverFactory(solver_name, solver_io="lp")
        solver.options["ResultFile"] = str(dir_str.output_resolve_dir / "infeasibility.ilp")
        # TODO: Move solver name to attributes.csv file?
    else:
        solver = SolverFactory(solver_name)
        if solver_name == "amplxpress":
            solver.options["IIS"] = ""
            # Create an 'iis' (Irreducible Infeasible Set) suffix component on the instance.
            # If the solver supports suffixes, Pyomo will receive the stored information.
            # This should work for any AMPL-interfaced solver, but the only one we use that way
            # currently is XPRESS.
            resolve_model.model.iis = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Read solver settings from ResolveCase instance
    for s, options in resolve_model.solver_options.items():
        if s == solver_name:
            for option, value in options.items():
                solver.options[option] = value

    # Start solver (which will write out LP file, then call solver executable)
    logger.info("Writing problem file & starting solver...")

    debug = log_level == "DEBUG"
    solution = solver.solve(
        resolve_model.model, keepfiles=debug, tee=True, symbolic_solver_labels=symbolic_solver_labels
    )

    # If no integer variables, this will be an instance of `pyomo.opt.results.container.UndefinedData`
    if isinstance(solution.Problem._list[0]["Number of integer variables"], int):
        if solution.Problem._list[0]["Number of integer variables"] > 0:
            logger.info("Fixing integer variables and re-solving...")
            # I'm lazy, but this could be more robust
            for idx in resolve_model.model.Integer_Build:
                resolve_model.model.Integer_Build[idx].fix()

            solution = solver.solve(
                resolve_model.model, keepfiles=debug, tee=True, symbolic_solver_labels=symbolic_solver_labels
            )

    # escape if model is infeasible
    if solution.solver.termination_condition == TerminationCondition.infeasible:
        if solver_name in {"gurobi", "cplex"} and not symbolic_solver_labels:
            if not debug:
                logger.warning(
                    "Model was infeasible; re-solving model with `symbolic_solver_labels=True` to provide better Irreducible Infeasible Set (IIS) information."
                )
                solver.solve(resolve_model.model, keepfiles=True, tee=True, symbolic_solver_labels=True)

            raise RuntimeError(
                f"Model was infeasible; check ILP file for infeasible constraints {(dir_str.output_resolve_dir / 'infeasibility.ilp').absolute()}"
            )

        elif solver_name == "amplxpress":
            raise RuntimeError(
                "Solver identified the following constraints to be infeasible: \n"
                + "\n  ".join(sorted(c.name for c in resolve_model.model.iis))
            )
        else:
            raise RuntimeError(
                "Model was infeasible. Use CPLEX, Gurobi, or XPRESS (AMPL) to get better Irreducible Infeasible Set (IIS) information."
            )

    return solution


def get_objective_function_value(instance, output_dir: pathlib.Path):
    """Save the objective function value.

    Args:
        instance (pyo.ConcreteModel): Filled abstract model instance to solve.
        scenario_results_directory ([type]): Scenario results directory to write to.
    """
    logger.info("Objective function value is: {:,.2f}".format(instance.Total_Cost()))
    with open(output_dir / "objective_function_value.txt", "w") as writer:
        writer.write(f"Objective function value is: {str(instance.Total_Cost())}")


def _run_case(
    dir_str: DirStructure,
    resolve_settings_name: str,
    **kwargs
):
    """Initialize and run a Resolve case"""
    logger.debug(f"=== Starting _run_case ===")
    logger.debug(f"Base directory: {dir_str.data_dir}")
    logger.debug(f"Case name: {resolve_settings_name}")
    
    # Load cases_to_run.csv
    cases_path = dir_str.resolve_settings_dir / "cases_to_run.csv"
    logger.debug(f"Loading cases from: {cases_path}")
    cases_df = pd.read_csv(cases_path)
    
    case_row = cases_df[cases_df.case_name == resolve_settings_name].iloc[0]
    logger.debug(f"Selected case: {case_row.to_dict()}")
    
    # Create data dictionary
    data = {
        "dir_structure": dir_str,
        "system": case_row["system"],
        "name": resolve_settings_name,
        "case_name": resolve_settings_name
    }
    logger.debug(f"Data dictionary: {data}")
    
    # Get correct attributes.csv path
    attributes_path = dir_str.resolve_settings_dir / resolve_settings_name / "attributes.csv"
    logger.debug(f"Attributes file path: {attributes_path}")
    
    if not attributes_path.exists():
        raise FileNotFoundError(f"Attributes file not found at: {attributes_path}")
    
    _, resolve_model = model_formulation.ResolveCase.from_csv(
        filename=attributes_path,
        data=data,
        return_type=tuple,
        name=resolve_settings_name
    )
    return resolve_model


def main(
    resolve_settings_name: Optional[str] = typer.Argument(
        None,
        help="Name of a RESOLVE case (under ./data-tpp/settings/resolve). If `None`, will run all cases listed in ./data-tpp/settings/resolve/cases_to_run.csv",
    ),
    data_folder: str = typer.Option(
        "data-tpp",
        help="Path to data folder containing settings and other data files",
    ),
    solver_name: str = typer.Option(
        "appsi_highs",
        help="Name of the solver to use. See Pyomo documentation for solver names.",
    ),
    symbolic_solver_labels: bool = typer.Option(False, help="use symbolic solver labels"),
    log_json: bool = typer.Option(False, help="Serialize logging infromation as JSON"),
    log_level: str = typer.Option(
        "INFO",
        help="Any Python logging level: [DEBUG, INFO, WARNING, ERROR, CRITICAL]. "
        "Choosing DEBUG will also enable Pyomo `tee=True` and `symbolic_solver_labels` options.",
    ),
    extras: Optional[str] = typer.Option(
        "cpuc_irp", help="Enables a RESOLVE 'extras' module, which contains project-specific add-on constraints."
    ),
    raw_results: bool = typer.Option(
        False,
        help="If this option is passed, the model will report all Pyomo model components directly.",
    ),
    return_cases: bool = typer.Option(
        False, help="Whether or not to return a list of the completed cases when finished."
    ),
    raise_on_error: bool = typer.Option(
        False,
        help="Whether or not to raise an exception if one occurs during running of cases. Note that if you are running "
        "multiple cases, any cases subsequent to the raised exception will not run.",
    ),
    # TODO (2022-02-22): This should be restricted to only "approved" extras
) -> Optional[list[ResolveCase]]:
    logger.info(f"Resolve version: {__version__}")
    
    # Create folder structure using absolute path to data-tpp
    data_path = pathlib.Path(data_folder).absolute()
    logger.debug(f"Using data folder: {data_path}")
    
    dir_str = DirStructure(data_folder=data_path)
    if resolve_settings_name:
        cases_to_run = [resolve_settings_name]
    else:
        cases_to_run = pd.read_csv(dir_str.data_settings_dir / "resolve" / "cases_to_run.csv").iloc[:, 0].to_list()

    resolve_cases = []
    for resolve_settings_name in cases_to_run:
        logger.info(f"Loading Resolve case: {resolve_settings_name}")
        # Remove default loguru logger to stderr
        logger.remove()
        # Set stdout logging level
        logger.add(sys.__stdout__, level=log_level, serialize=log_json)

        # Make folders
        dir_str.make_resolve_dir(resolve_settings_name=resolve_settings_name, log_level=log_level)
        TempfileManager.tempdir = dir_str.output_resolve_dir

        if raise_on_error:
            resolve_model = _run_case(
                dir_str=dir_str,
                resolve_settings_name=resolve_settings_name,
                extras=extras,
                solver_name=solver_name,
                log_level=log_level,
                symbolic_solver_labels=symbolic_solver_labels,
                raw_results=raw_results,
            )
            if return_cases:
                resolve_cases.append(resolve_model)

        else:
            try:
                resolve_model = _run_case(
                    dir_str=dir_str,
                    resolve_settings_name=resolve_settings_name,
                    extras=extras,
                    solver_name=solver_name,
                    log_level=log_level,
                    symbolic_solver_labels=symbolic_solver_labels,
                    raw_results=raw_results,
                )

                if return_cases:
                    resolve_cases.append(resolve_model)

            except Exception as e:
                logger.error(f"Case {resolve_settings_name} failed. See error traceback below:")
                logger.error(traceback.format_exc())

        shutil.copytree(
            dir_str.resolve_passthrough_inputs, dir_str.output_resolve_dir / "passthrough", dirs_exist_ok=True
        )  # Fine

        logger.info("Done.")

    return resolve_cases if return_cases else None


if __name__ == "__main__":
    from rich.traceback import install

    install()

    try:
        typer.run(main)
    except ImportError:
        main()
