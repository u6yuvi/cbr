import gradio as gr
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import io
from PIL import Image
import base64
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import logging
import os
import time
import pickle
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:8000")

class CbRGradioApp:
    def __init__(self):
        self.tenant_id = None
        
    def create_tenant(self, name: str = "") -> str:
        """Create a new tenant."""
        try:
            url = f"{API_BASE_URL}/tenants"
            data = {"name": name} if name else {}
            response = requests.post(url, json=data)
            result = response.json()
            return f"Created tenant with ID: {result['tenant_id']}"
        except Exception as e:
            return f"Error creating tenant: {str(e)}"

    def list_tenants(self) -> str:
        """List all tenants."""
        try:
            url = f"{API_BASE_URL}/tenants"
            response = requests.get(url)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error listing tenants: {str(e)}"

    def set_tenant(self, tenant_id: str) -> str:
        """Set the current tenant ID."""
        self.tenant_id = tenant_id
        return f"Set current tenant ID to: {tenant_id}"

    def get_model_info(self) -> str:
        """Get model information for current tenant."""
        if not self.tenant_id:
            return "Please set a tenant ID first"
        try:
            url = f"{API_BASE_URL}/model/info"
            headers = {"X-Tenant-ID": self.tenant_id}
            response = requests.get(url, headers=headers)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error getting model info: {str(e)}"

    def add_class(self, class_name: str, images: List[Path]) -> str:
        """Add a new class with examples."""
        if not self.tenant_id:
            return "Please set a tenant ID first"
        if not class_name:
            return "Please provide a class name"
        if not images:
            return "Please provide at least one image"
            
        try:
            url = f"{API_BASE_URL}/class/add/{class_name}"
            headers = {"X-Tenant-ID": self.tenant_id}
            files = [("files", open(img, "rb")) for img in images]
            response = requests.post(url, headers=headers, files=files)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error adding class: {str(e)}"
        finally:
            for _, f in files:
                f.close()

    def update_class(self, class_name: str, images: List[Path], append: bool) -> str:
        """Update or append examples to a class."""
        if not self.tenant_id:
            return "Please set a tenant ID first"
        if not class_name:
            return "Please provide a class name"
        if not images:
            return "Please provide at least one image"
            
        try:
            url = f"{API_BASE_URL}/class/update/{class_name}"
            headers = {"X-Tenant-ID": self.tenant_id}
            files = [("files", open(img, "rb")) for img in images]
            data = {"append": "true" if append else "false"}
            response = requests.post(url, headers=headers, files=files, data=data)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error updating class: {str(e)}"
        finally:
            for _, f in files:
                f.close()

    def remove_class(self, class_name: str) -> str:
        """Remove a class."""
        if not self.tenant_id:
            return "Please set a tenant ID first"
        if not class_name:
            return "Please provide a class name"
            
        try:
            url = f"{API_BASE_URL}/class/{class_name}"
            headers = {"X-Tenant-ID": self.tenant_id}
            response = requests.delete(url, headers=headers)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error removing class: {str(e)}"

    def predict(self, images: List[Path]) -> tuple[List[Image.Image], str]:
        """Make predictions on one or more images."""
        if not self.tenant_id:
            return [], "Please set a tenant ID first"
        if not images:
            return [], "Please provide at least one image"
            
        try:
            url = f"{API_BASE_URL}/predict"
            headers = {"X-Tenant-ID": self.tenant_id}
            
            results = []
            pil_images = []
            for image in images:
                files = {"file": open(image, "rb")}
                try:
                    response = requests.post(url, headers=headers, files=files)
                    result = response.json()
                    results.append(result)
                    # Load image for display
                    pil_images.append(Image.open(image))
                finally:
                    files["file"].close()
            
            # Format results as markdown
            markdown_results = "### Prediction Results\n\n"
            for idx, result in enumerate(results):
                markdown_results += f"**Image {idx + 1}**:\n"
                markdown_results += f"- Predicted Class: {result.get('predicted_class', 'N/A')}\n"
                markdown_results += f"- Confidence: {result.get('confidence', 'N/A')}\n\n"
                    
            return pil_images, markdown_results
        except Exception as e:
            return [], f"Error making prediction: {str(e)}"

    def get_class_images(self, class_name: str) -> List[Image.Image]:
        """Get all images for a specific class."""
        if not self.tenant_id:
            return []
        if not class_name:
            return []
            
        try:
            # First get model info to check if class exists
            url = f"{API_BASE_URL}/model/info"
            headers = {"X-Tenant-ID": self.tenant_id}
            response = requests.get(url, headers=headers)
            model_info = response.json()
            
            if class_name not in model_info["available_classes"]:
                return []
            
            # Get images for the class
            url = f"{API_BASE_URL}/class/{class_name}/images"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Convert base64 to PIL Images
                images = []
                image_bytes_list = response.json()["images"]
                for img_b64 in image_bytes_list:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)
                return images
            return []
        except Exception as e:
            print(f"Error getting class images: {str(e)}")
            return []

    def predict_single(self, image: Path) -> Dict:
        """Make prediction on a single image."""
        if not self.tenant_id:
            return {"error": "Please set a tenant ID first"}
        if not image:
            return {"error": "Please provide an image"}
            
        try:
            url = f"{API_BASE_URL}/predict"
            headers = {"X-Tenant-ID": self.tenant_id}
            files = {"file": open(image, "rb")}
            try:
                response = requests.post(url, headers=headers, files=files)
                result = response.json()
                return result
            finally:
                files["file"].close()
        except Exception as e:
            return {"error": str(e)}

    def calculate_metrics(self, true_labels: List[str], predicted_labels: List[str]) -> Dict:
        """Calculate various performance metrics."""
        try:
            metrics = {
                "accuracy": accuracy_score(true_labels, predicted_labels),
                "f1": f1_score(true_labels, predicted_labels, average='weighted'),
                "recall": recall_score(true_labels, predicted_labels, average='weighted'),
                "precision": precision_score(true_labels, predicted_labels, average='weighted')
            }
            return metrics
        except Exception as e:
            return {"error": f"Error calculating metrics: {str(e)}"}

    def format_metrics(self, metrics: Dict) -> str:
        """Format metrics as markdown."""
        if "error" in metrics:
            return f"### Error\n{metrics['error']}"
            
        return f"""### Performance Metrics
- Accuracy: {metrics['accuracy']:.2%}
- F1 Score: {metrics['f1']:.2%}
- Recall: {metrics['recall']:.2%}
- Precision: {metrics['precision']:.2%}
"""

def create_gradio_interface():
    app = CbRGradioApp()
    
    # Setup session directory
    SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_data")
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    def create_new_session():
        """Create a new unique session ID"""
        session_id = f"user_session_{int(time.time())}_{os.urandom(4).hex()}"
        session_file = os.path.join(SESSION_DIR, f"{session_id}.pkl")
        
        # Initialize session data
        session_data = {
            "true_labels": [],
            "pred_labels": [],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to disk
        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)
            
        return session_id, f"Created new session: {session_id}"
    
    def load_session(session_id):
        """Load an existing session by ID"""
        if not session_id:
            return session_id, "Please create or enter a valid session ID"
            
        session_file = os.path.join(SESSION_DIR, f"{session_id}.pkl")
        if not os.path.exists(session_file):
            return session_id, f"Session {session_id} not found"
            
        return session_id, f"Loaded session: {session_id}"
    
    def save_session_data(session_id, data):
        """Save session data to disk"""
        if not session_id:
            return
            
        session_file = os.path.join(SESSION_DIR, f"{session_id}.pkl")
        data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(session_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_session_data(session_id):
        """Load session data from disk"""
        if not session_id:
            return None
            
        session_file = os.path.join(SESSION_DIR, f"{session_id}.pkl")
        if not os.path.exists(session_file):
            return None
            
        with open(session_file, 'rb') as f:
            return pickle.load(f)
    
    with gr.Blocks(title="Classification by Retrieval (CbR) Interface") as interface:
        gr.Markdown("# Classification by Retrieval (CbR) Interface")
        
        # Session management at the top level
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### User Session")
                create_session_btn = gr.Button("Create New Session")
                
            with gr.Column(scale=2):
                # Using a regular textbox instead of gr.State
                session_id_textbox = gr.Textbox(label="Session ID", placeholder="Enter an existing session ID or create a new one")
                load_session_btn = gr.Button("Load Session")
                
            with gr.Column(scale=1):
                session_status = gr.Markdown("")
                
        # Wire up session management 
        create_session_btn.click(
            fn=create_new_session,
            inputs=[],
            outputs=[session_id_textbox, session_status]
        )
        
        load_session_btn.click(
            fn=load_session,
            inputs=[session_id_textbox],
            outputs=[session_id_textbox, session_status]
        )
        
        with gr.Tab("Tenant Management"):
            with gr.Row():
                tenant_name = gr.Textbox(label="Tenant Name (optional)")
                create_tenant_btn = gr.Button("Create Tenant")
            create_tenant_output = gr.Textbox(label="Create Tenant Result")
            create_tenant_btn.click(app.create_tenant, inputs=[tenant_name], outputs=create_tenant_output)
            
            list_tenants_btn = gr.Button("List Tenants")
            list_tenants_output = gr.Textbox(label="Tenants List")
            list_tenants_btn.click(app.list_tenants, inputs=[], outputs=list_tenants_output)
            
            with gr.Row():
                tenant_id = gr.Textbox(label="Tenant ID")
                set_tenant_btn = gr.Button("Set Current Tenant")
            set_tenant_output = gr.Textbox(label="Set Tenant Result")
            set_tenant_btn.click(app.set_tenant, inputs=[tenant_id], outputs=set_tenant_output)
        
        with gr.Tab("Model Management"):
            with gr.Row():
                model_info_btn = gr.Button("Get Model Info")
            model_info_output = gr.Textbox(label="Model Information")
            model_info_btn.click(app.get_model_info, inputs=[], outputs=model_info_output)
            
            gr.Markdown("### Add New Class")
            with gr.Row():
                add_class_name = gr.Textbox(label="Class Name")
                add_class_images = gr.File(label="Example Images", file_count="multiple")
            add_class_btn = gr.Button("Add Class")
            add_class_output = gr.Textbox(label="Add Class Result")
            add_class_btn.click(app.add_class, inputs=[add_class_name, add_class_images], outputs=add_class_output)
            
            gr.Markdown("### Update Class")
            with gr.Row():
                update_class_name = gr.Textbox(label="Class Name")
                update_class_images = gr.File(label="Example Images", file_count="multiple")
                update_append = gr.Checkbox(label="Append Mode")
            update_class_btn = gr.Button("Update Class")
            update_class_output = gr.Textbox(label="Update Class Result")
            update_class_btn.click(app.update_class, inputs=[update_class_name, update_class_images, update_append], outputs=update_class_output)
            
            gr.Markdown("### Remove Class")
            with gr.Row():
                remove_class_name = gr.Textbox(label="Class Name")
                remove_class_btn = gr.Button("Remove Class")
            remove_class_output = gr.Textbox(label="Remove Class Result")
            remove_class_btn.click(app.remove_class, inputs=[remove_class_name], outputs=remove_class_output)
        
        with gr.Tab("Prediction"):
            with gr.Row():
                predict_image = gr.File(label="Upload Image(s)", file_count="multiple")
                predict_btn = gr.Button("Make Prediction")
            
            with gr.Row():
                with gr.Column():
                    result_gallery = gr.Gallery(
                        label="Predicted Images",
                        show_label=True,
                        elem_id="result_gallery",
                        columns=[2],
                        rows=[2],
                        height="auto",
                        allow_preview=True
                    )
                with gr.Column():
                    predict_output = gr.Markdown()
            
            predict_btn.click(
                app.predict,
                inputs=[predict_image],
                outputs=[result_gallery, predict_output]
            )
        
        with gr.Tab("View Class Images"):
            with gr.Row():
                view_class_name = gr.Textbox(label="Class Name")
                view_images_btn = gr.Button("View Images")
            
            gallery = gr.Gallery(
                label="Class Images",
                show_label=True,
                elem_id="gallery",
                columns=[3], 
                rows=[2],
                height="auto",
                allow_preview=True
            )
            
            view_images_btn.click(
                app.get_class_images,
                inputs=[view_class_name],
                outputs=[gallery]
            )
        
        with gr.Tab("Progressive Learning"):
            gr.Markdown("### Progressive Learning")
            gr.Markdown("Upload an image, get a prediction, and provide feedback.")
            
            # Display current session info
            session_info = gr.Markdown("Please create or load a session using the controls above.")
            
            # Update session display button
            def update_session_display(session_id):
                if not session_id:
                    return "No active session. Please create or load a session above."
                return f"Using session: {session_id}"
                
            update_session_btn = gr.Button("Show Current Session")
            update_session_btn.click(
                fn=update_session_display,
                inputs=[session_id_textbox],
                outputs=[session_info]
            )
            
            # Upload section - one image at a time
            upload_file = gr.File(label="Upload Image", file_count="single")
            predict_btn = gr.Button("Get Prediction")
            
            # Display section
            image_display = gr.Image(
                label="Image", 
                type="pil",
                height=200,  # Smaller height
                width=200   # Smaller width
            )
            prediction_text = gr.Markdown()
            
            # Available Classes Reference Section
            gr.Markdown("### Available Classes for Reference")
            available_classes_display = gr.Markdown("Click 'Show Available Classes' to see the list")
            show_classes_btn = gr.Button("Show Available Classes")
            
            def format_available_classes():
                """Get and format available classes for display"""
                if not app.tenant_id:
                    return "Please set a tenant ID first"
                try:
                    url = f"{API_BASE_URL}/model/info"
                    headers = {"X-Tenant-ID": app.tenant_id}
                    response = requests.get(url, headers=headers)
                    if response.status_code != 200:
                        return "Error fetching classes. Please check your tenant ID."
                    
                    model_info = response.json()
                    classes = model_info.get("available_classes", [])
                    
                    if not classes:
                        return "No classes available. Please add some classes in the Model Management tab."
                    
                    # Format classes as a nice markdown list
                    class_list = "\n".join([f"- `{cls}`" for cls in sorted(classes)])
                    return f"**Available Classes:**\n\n{class_list}"
                except Exception as e:
                    return f"Error: {str(e)}"
            
            show_classes_btn.click(
                fn=format_available_classes,
                inputs=[],
                outputs=[available_classes_display]
            )
            
            # Feedback section
            gr.Markdown("### Provide Feedback")
            feedback_text = gr.Textbox(label="Enter correct class label", placeholder="Type the correct class name...")
            submit_btn = gr.Button("Submit Feedback")
            feedback_result = gr.Markdown()
            
            # Simple storage for current prediction - using visible textbox for debugging
            current_pred_class = gr.Textbox(label="Current Prediction (System Use)", visible=True)
            
            def get_prediction(file, session_id):
                if not session_id:
                    return None, "Please create or load a session first", ""
                    
                if not file:
                    return None, "Please upload an image", ""
                
                try:
                    # Make prediction
                    result = app.predict_single(file)
                    if "error" in result:
                        return None, f"Error: {result['error']}", ""
                    
                    # Get prediction class
                    pred_class = result.get('predicted_class', 'unknown')
                    
                    # Display image and prediction
                    return (Image.open(file), 
                            f"""### Prediction Result
- Predicted Class: {pred_class}
- Confidence: {result.get('confidence', 'N/A')}

Please provide the correct class label:""",
                            pred_class)
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    return None, f"Error: {str(e)}", ""
            
            def submit_feedback(label, file, pred_class, session_id):
                logger.info(f"Submit feedback: label={label}, pred_class={pred_class}, session={session_id}")
                if not session_id:
                    return "Please create or load a session first"
                    
                if not label.strip():
                    return "Please provide a class label"
                if not file:
                    return "No image is currently uploaded"
                if not pred_class:
                    return "No prediction available"
                
                try:
                    # Update the class with feedback
                    app.update_class(label.strip(), [file], True)
                    
                    # Load current session data
                    data = load_session_data(session_id)
                    if not data:
                        return f"Error: Could not load session data for {session_id}"
                    
                    # Update metrics
                    data["true_labels"].append(label.strip())
                    data["pred_labels"].append(pred_class)
                    
                    # Save updated data
                    save_session_data(session_id, data)
                    
                    return f"Successfully added image to class '{label.strip()}'. Data saved to session {session_id}."
                except Exception as e:
                    logger.error(f"Feedback error: {str(e)}")
                    return f"Error updating class: {str(e)}"
            
            # Connect the components
            predict_btn.click(
                fn=get_prediction,
                inputs=[upload_file, session_id_textbox],
                outputs=[image_display, prediction_text, current_pred_class]
            )
            
            submit_btn.click(
                fn=submit_feedback,
                inputs=[feedback_text, upload_file, current_pred_class, session_id_textbox],
                outputs=[feedback_result]
            )
        
        with gr.Tab("Model Performance"):
            gr.Markdown("### Model Performance Metrics")
            gr.Markdown("View the cumulative performance metrics from your progressive learning session.")
            
            # Display current session info for metrics
            performance_session_info = gr.Markdown("Please create or load a session using the controls above.")
            
            # Update session display button for metrics
            update_perf_session_btn = gr.Button("Show Current Session")
            update_perf_session_btn.click(
                fn=update_session_display,  # Reuse the same function
                inputs=[session_id_textbox],
                outputs=[performance_session_info]
            )
            
            def calculate_metrics(session_id):
                """Calculate metrics from session data"""
                if not session_id:
                    return "Please create or load a session first to view metrics."
                
                try:
                    # Load session data
                    data = load_session_data(session_id)
                    if not data:
                        return f"Error: Could not load session data for {session_id}"
                        
                    true_labels = data["true_labels"]
                    pred_labels = data["pred_labels"]
                    last_update = data["last_update"]
                    
                    if not true_labels or not pred_labels:
                        return f"""### No Metrics Available
No data collected in session {session_id}.
Start using the Progressive Learning tab to accumulate performance metrics."""
                    
                    # Calculate metrics
                    metrics = app.calculate_metrics(true_labels, pred_labels)
                    
                    # Format class distribution
                    from collections import Counter
                    counts = Counter(true_labels)
                    total = len(true_labels)
                    
                    class_dist = ""
                    for class_name, count in sorted(counts.items()):
                        percentage = (count / total) * 100
                        class_dist += f"- {class_name}: {count} samples ({percentage:.1f}%)\n"
                    
                    return f"""### Cumulative Performance Metrics
#### Session ID: {session_id}
#### Last Updated: {last_update}

- Total Samples: {total}
- Accuracy: {metrics['accuracy']:.2%}
- F1 Score: {metrics['f1']:.2%}
- Recall: {metrics['recall']:.2%}
- Precision: {metrics['precision']:.2%}

#### Class Distribution
{class_dist}"""
                except Exception as e:
                    logger.error(f"Metrics calculation error: {str(e)}")
                    return f"Error calculating metrics: {str(e)}"
            
            # Add refresh button for metrics
            metrics_display = gr.Markdown()
            refresh_btn = gr.Button("Calculate Metrics")
            
            # Button to reset metrics
            def reset_metrics(session_id):
                """Reset metrics for the current session"""
                if not session_id:
                    return "Please create or load a session first"
                    
                data = load_session_data(session_id)
                if not data:
                    return f"Error: Could not load session data for {session_id}"
                    
                data["true_labels"] = []
                data["pred_labels"] = []
                save_session_data(session_id, data)
                return f"Metrics for session {session_id} have been reset."
            
            reset_btn = gr.Button("Reset Metrics")
            reset_result = gr.Markdown()
            
            # List available sessions
            def list_sessions():
                """List all available sessions"""
                sessions = []
                for file in os.listdir(SESSION_DIR):
                    if file.endswith(".pkl"):
                        session_id = file[:-4]  # Remove .pkl extension
                        session_path = os.path.join(SESSION_DIR, file)
                        try:
                            with open(session_path, 'rb') as f:
                                data = pickle.load(f)
                            
                            last_update = data.get("last_update", "Unknown")
                            num_samples = len(data.get("true_labels", []))
                            
                            sessions.append({
                                "session_id": session_id,
                                "last_update": last_update,
                                "samples": num_samples
                            })
                        except:
                            continue
                
                if not sessions:
                    return "No sessions found."
                
                result = "### Available Sessions\n\n"
                for s in sorted(sessions, key=lambda x: x["last_update"], reverse=True):
                    result += f"- **{s['session_id']}**: {s['samples']} samples, last updated {s['last_update']}\n"
                
                return result
            
            sessions_list = gr.Markdown()
            list_sessions_btn = gr.Button("List All Sessions")
            
            # Connect buttons
            refresh_btn.click(
                fn=calculate_metrics,
                inputs=[session_id_textbox],
                outputs=[metrics_display]
            )
            
            reset_btn.click(
                fn=reset_metrics,
                inputs=[session_id_textbox],
                outputs=[reset_result]
            )
            
            list_sessions_btn.click(
                fn=list_sessions,
                inputs=[],
                outputs=[sessions_list]
            )
    
    return interface

if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface...")
        interface = create_gradio_interface()
        logger.info("Gradio interface created successfully")
        
        # Get server settings from environment or use defaults
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        
        logger.info(f"Launching Gradio server on {server_name}:{server_port}")
        # interface.launch(
        #     server_name=server_name,
        #     server_port=server_port,
        #     share=False,  # Enable sharing to handle Docker networking
        #     # debug=True,
        #     # show_error=True,
        #     # quiet=False,
        #     # allowed_paths=["."],
        #     # root_path="",
        #     # inbrowser=False,
        #     # favicon_path=None,
        #     # ssl_verify=False,
        #     # ssl_certfile=None,
        #     # ssl_keyfile=None,
        #     # ssl_keyfile_password=None,
        #     # show_api=False,
        #     # max_threads=40,
        #     # auth=None,
        #     # auth_message=None,
        #     # prevent_thread_lock=False
        # )
        interface.launch(server_name=server_name, server_port=server_port, share=False)
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {str(e)}")
        raise 