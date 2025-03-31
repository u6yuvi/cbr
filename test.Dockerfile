# Step 1: Use a base Python image
FROM python:3.12-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the application files (app.py) into the container

# Step 4: Install dependencies
RUN pip install --no-cache-dir gradio scikit-learn

COPY test_gradio.py /app

# Step 5: Expose the port the app will run on
EXPOSE 7860

# Step 6: Define the command to run the Gradio app when the container starts
CMD ["python", "test_gradio.py"]