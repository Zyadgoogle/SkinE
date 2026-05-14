# 🌟 SkinE AI - Python Backend API

This repository contains the backend AI engine for the **SkinE** web application. It acts as a REST API that receives user photos, analyzes them using a custom PyTorch Vision Model, and generates personalized skincare routines using LangChain and Llama 3 (via Groq).

## 🚀 Features
* **Face Detection & Cropping:** Automatically detects and isolates the user's face from the uploaded image using `face_recognition` and OpenCV.
* **Skin Type Classification:** Uses a trained PyTorch EfficientNet-B0 model (`skin_model.pth`) to classify skin into 4 types: Dry, Normal, Oily, or Combination.
* **AI Dermatologist Recommendations:** Uses Llama 3 to generate structured JSON routines, ingredient guides, and strictly categorizes real-world product recommendations into **Affordable (Drugstore)** and **High-End (Luxury)** options.
* **CORS Enabled:** Ready to seamlessly communicate with external frontends (React, Vanilla JS, Supabase, etc.).

---

## 🛠️ Project Structure
* `app.py`: The main Flask server that handles HTTP requests, CORS, and image processing.
* `recommendation.py`: The LangChain engine that enforces the JSON output structure and talks to the Groq API.
* `client.py`: A local terminal-based testing client (Frontend developers don't need to run this).
* `skin_model.pth`: The trained PyTorch weights for the EfficientNet vision model *(Make sure this is uploaded!)*.

