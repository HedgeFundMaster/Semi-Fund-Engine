# Semi-Liquid Engine

A simple web application that uses OpenAI's GPT-2 model to generate text based on a user's prompt. This project is built with Streamlit, providing a straightforward and interactive user interface.

## Features

-   **Text Generation**: Leverages the power of GPT-2 to generate creative and coherent text.
-   **Simple UI**: A clean and simple user interface built with Streamlit.
-   **Easy Setup**: A script is provided to generate the necessary secrets file.

## Requirements

-   Python 3.7+
-   An OpenAI API key

The project dependencies are listed in the `requirements.txt` file:
-   `streamlit`
-   `openai`
-   `toml`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd semi_liquid_engine
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the secrets file:**
    Run the `generate_secrets.py` script to create the `.streamlit/secrets.toml` file.
    ```bash
    python generate_secrets.py
    ```

5.  **Add your OpenAI API key:**
    Open the newly created `.streamlit/secrets.toml` file and replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.

## Usage

To run the application, use the following command:
```bash
streamlit run streamlit_app.py
```
Now, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Testing

The `test.py` file is currently empty. Contributions to add tests for the application are welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
