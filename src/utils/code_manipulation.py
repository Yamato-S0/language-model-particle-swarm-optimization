import multiprocessing
import time


def add_code_wrapper(code: str) -> str:
    """Wrap code in a function definition."""
    return f"```python\n{code}\n```"


def delete_code_wrapper(code: str) -> str:
    """Delete code wrapper."""
    return code.replace("```python", "").replace("```", "")


def run_with_timeout(func, args=(), kwargs=None, timeout=5):
    """
    Runs a function with a timeout using multiprocessing. If the function
    does not complete within the given timeout, a TimeoutError is raised.

    Args:
        func: The function to run.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        timeout: Maximum time (in seconds) to allow the function to run.

    Returns:
        The result of the function if it completes in time.

    Raises:
        TimeoutError: If the function execution exceeds the timeout.
    """
    if kwargs is None:
        kwargs = {}

    # Define a wrapper to run the function
    def target(queue):
        try:
            result = func(*args, **kwargs)
            queue.put((result, None))  # Store result in the queue
        except Exception as e:
            queue.put((None, e))  # Store exception in the queue

    # Create a queue for communication
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue,))
    process.start()
    process.join(timeout)  # Wait for the process to complete or timeout

    if process.is_alive():
        process.terminate()  # Forcefully terminate the process
        process.join()  # Ensure cleanup
        raise TimeoutError(f"Function execution exceeded {timeout} seconds")

    # Retrieve the result or exception from the queue
    result, exception = queue.get()
    if exception:
        raise exception

    return result


# Usage example
def example_function(duration):
    time.sleep(duration)
    return f"Ran for {duration} seconds"


try:
    result = run_with_timeout(example_function, args=(10,), timeout=3)
    print(result)
except TimeoutError as e:
    print(e)
