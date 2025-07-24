use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use ndarray::Array2;
use image::{GrayImage, ImageBuffer};
use nn_lib::sequential::Sequential;

#[derive(Deserialize)]
pub struct PredictionRequest {
    pub canvas_data: CanvasData,
    pub model_type: String, // "mlp" or "cnn"
}

#[derive(Deserialize)]
pub struct CanvasData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA data from canvas
}

#[derive(Serialize)]
pub struct PredictionResponse {
    pub predictions: Vec<f64>,
    pub predicted_digit: usize,
    pub confidence: f64,
    pub model_used: String,
}

pub struct WebSocketServer {
    mlp_model: Arc<Mutex<Sequential>>,
    cnn_model: Arc<Mutex<Option<Sequential>>>,
}

impl WebSocketServer {
    pub fn new(mlp_model: Sequential, cnn_model: Option<Sequential>) -> Self {
        Self {
            mlp_model: Arc::new(Mutex::new(mlp_model)),
            cnn_model: Arc::new(Mutex::new(cnn_model)),
        }
    }

    pub async fn start(&self, addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("WebSocket server listening on: {}", addr);

        while let Ok((stream, _)) = listener.accept().await {
            let mlp_model = Arc::clone(&self.mlp_model);
            let cnn_model = Arc::clone(&self.cnn_model);
            
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, mlp_model, cnn_model).await {
                    eprintln!("Error handling connection: {}", e);
                }
            });
        }
        Ok(())
    }

    async fn handle_connection(
        stream: tokio::net::TcpStream,
        mlp_model: Arc<Mutex<Sequential>>,
        cnn_model: Arc<Mutex<Option<Sequential>>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ws_stream = accept_async(stream).await?;
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        while let Some(msg) = ws_receiver.next().await {
            match msg? {
                Message::Text(text) => {
                    if let Ok(request) = serde_json::from_str::<PredictionRequest>(&text) {
                        let prediction = Self::process_prediction_request(
                            request,
                            &mlp_model,
                            &cnn_model,
                        ).await?;
                        
                        let response = serde_json::to_string(&prediction)?;
                        ws_sender.send(Message::Text(response)).await?;
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        Ok(())
    }

    async fn process_prediction_request(
        request: PredictionRequest,
        mlp_model: &Arc<Mutex<Sequential>>,
        cnn_model: &Arc<Mutex<Option<Sequential>>>,
    ) -> Result<PredictionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let canvas_data = request.canvas_data;
        let use_cnn = request.model_type == "cnn";
        // Convert RGBA canvas data to grayscale
        let mut gray_pixels = Vec::new();
        for chunk in canvas_data.pixels.chunks(4) {
            let r = chunk[0] as f32;
            let g = chunk[1] as f32;
            let b = chunk[2] as f32;
            let a = chunk[3] as f32;
            
            // Convert to grayscale and apply alpha
            let gray = ((0.299 * r + 0.587 * g + 0.114 * b) * (a / 255.0)) as u8;
            gray_pixels.push(gray);
        }

        // Create grayscale image
        let img: GrayImage = ImageBuffer::from_raw(
            canvas_data.width,
            canvas_data.height,
            gray_pixels,
        ).ok_or("Failed to create image from canvas data")?;

        // Resize to 28x28 like in the original GUI code
        let resized_img: GrayImage = image::imageops::resize(
            &img,
            28,
            28,
            image::imageops::FilterType::Lanczos3,
        );

        // Normalize pixels (0-1 range)
        let normalized_pixels: Vec<f64> = resized_img
            .pixels()
            .map(|p| p[0] as f64 / 255.0)
            .collect();

        let input_array = Array2::from_shape_vec((1, 28 * 28), normalized_pixels)?;
        let input = input_array.into_dyn();

        // Make prediction using selected model
        let (predictions, model_used) = if use_cnn {
            let cnn_guard = cnn_model.lock().await;
            if let Some(ref cnn) = *cnn_guard {
                (cnn.predict(&input)?, "cnn".to_string())
            } else {
                // Fallback to MLP if CNN not available
                let mlp = mlp_model.lock().await;
                (mlp.predict(&input)?, "mlp".to_string())
            }
        } else {
            let mlp = mlp_model.lock().await;
            (mlp.predict(&input)?, "mlp".to_string())
        };
        
        // Find the digit with highest confidence
        let mut max_confidence = 0.0;
        let mut predicted_digit = 0;
        let pred_vec: Vec<f64> = predictions.iter().cloned().collect();
        
        for (i, &confidence) in pred_vec.iter().enumerate() {
            if confidence > max_confidence {
                max_confidence = confidence;
                predicted_digit = i;
            }
        }

        Ok(PredictionResponse {
            predictions: pred_vec,
            predicted_digit,
            confidence: max_confidence,
            model_used,
        })
    }
}