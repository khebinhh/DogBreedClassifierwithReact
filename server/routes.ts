import type { Express, Request } from "express";
import multer from "multer";
import { db } from "../db";
import { classifications } from "@db/schema";
import { desc } from "drizzle-orm";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB limit
  },
});

interface MulterRequest extends Request {
  file?: Express.Multer.File;
}

export function registerRoutes(app: Express) {
  // Get classification history
  app.get("/api/classifications", async (req, res) => {
    try {
      const results = await db.query.classifications.findMany({
        orderBy: [desc(classifications.createdAt)],
        limit: 9,
      });
      res.json(results);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch classifications" });
    }
  });

  // Handle image upload and classification
  app.post("/api/classify", upload.single("image"), async (req: MulterRequest, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No image provided" });
    }

    try {
      // Upload image to a temporary URL (in production, use proper storage)
      const imageUrl = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;
      
      // Process the image using TensorFlow.js
      const tf = require('@tensorflow/tfjs-node');
      const model = await tf.loadGraphModel('https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5');
      
      // Preprocess image
      const image = await tf.node.decodeImage(req.file.buffer);
      const processedImage = tf.tidy(() => {
        return tf.image
          .resizeBilinear(image, [224, 224])
          .toFloat()
          .sub(127.5)
          .div(127.5)
          .expandDims();
      });
      
      // Get predictions
      const predictions = await model.predict(processedImage).data();
      const confidence = Math.max(...predictions);
      const breedIndex = predictions.indexOf(confidence);
      
      // Map to breed (simplified mapping)
      const breeds = ['Labrador', 'Golden Retriever', 'German Shepherd', 'Bulldog', 'Poodle'];
      const breed = breeds[breedIndex % breeds.length];
      
      // Store classification result
      const result = await db.insert(classifications).values({
        imageUrl,
        breed,
        confidence: confidence.toFixed(4),
      }).returning();
      
      // Cleanup
      tf.dispose([image, processedImage]);

      res.json(result[0]);
    } catch (error) {
      console.error('Classification error:', error);
      res.status(500).json({ error: "Classification failed" });
    }
  });
}
