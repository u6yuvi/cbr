import gradio as gr
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json
import io
from PIL import Image
import base64

# Constants
API_BASE_URL = "http://localhost:8000"

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

    def predict(self, image: Path) -> str:
        """Make a prediction on an image."""
        if not self.tenant_id:
            return "Please set a tenant ID first"
        if not image:
            return "Please provide an image"
            
        try:
            url = f"{API_BASE_URL}/predict"
            headers = {"X-Tenant-ID": self.tenant_id}
            files = {"file": open(image, "rb")}
            response = requests.post(url, headers=headers, files=files)
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error making prediction: {str(e)}"
        finally:
            files["file"].close()

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

def create_gradio_interface():
    app = CbRGradioApp()
    
    with gr.Blocks(title="Classification by Retrieval (CbR) Interface") as interface:
        gr.Markdown("# Classification by Retrieval (CbR) Interface")
        
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
                predict_image = gr.Image(label="Upload Image", type="filepath")
                predict_btn = gr.Button("Make Prediction")
            predict_output = gr.Textbox(label="Prediction Result")
            predict_btn.click(app.predict, inputs=[predict_image], outputs=predict_output)
        
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
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True) 