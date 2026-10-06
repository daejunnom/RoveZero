use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction(signature = (position_command, simulation_limit=128))]
fn search_once(
    py: Python<'_>,
    position_command: String,
    simulation_limit: u64,
) -> PyResult<Bound<'_, PyDict>> {
    // Owned String crosses detach; Rust never calls Python during the search.
    let measured = py
        .detach(move || rz_bindings_core::search_once(&position_command, simulation_limit))
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("best_move", measured.result.best_move)?;
    result.set_item("terminal", measured.result.terminal)?;
    result.set_item(
        "completed_simulations",
        measured.result.completed_simulations,
    )?;
    result.set_item("consumed_evaluations", measured.result.consumed_evaluations)?;
    let output = PyDict::new(py);
    output.set_item("result", result)?;
    output.set_item("initialization_ns", measured.initialization_ns)?;
    output.set_item("search_body_ns", measured.search_body_ns)?;
    output.set_item("shutdown_ns", measured.shutdown_ns)?;
    output.set_item("rust_call_ns", measured.rust_call_ns)?;
    Ok(output)
}

#[pymodule]
fn rz_bindings_poc(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(search_once, module)?)?;
    Ok(())
}
