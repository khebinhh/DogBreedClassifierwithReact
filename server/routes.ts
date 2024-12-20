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
  app.post("/api/classify", upload.single("file"), async (req: MulterRequest, res) => {
    if (!req.file) {
      console.error("Classification error: No file provided");
      return res.status(400).json({ error: "No image provided" });
    }

    try {
      console.log(`Processing image: ${req.file.originalname}, type: ${req.file.mimetype}, size: ${req.file.size}bytes`);
      
      // Upload image to a temporary URL (in production, use proper storage)
      const imageUrl = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;
      
      // Create form data for ML service
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append('file', blob, req.file.originalname);
      
      // Process the image using PyTorch service
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`ML service error: ${errorText}`);
        throw new Error(`Classification service failed: ${errorText}`);
      }

      const classificationResult = await response.json();
      console.log('Classification result:', classificationResult);
      
      // Store top prediction in database
      const topPrediction = classificationResult.predictions[0];
      await db.insert(classifications).values({
        imageUrl,
        breed: topPrediction.breed,
        confidence: topPrediction.confidence.toFixed(4),
      });

      // Send complete classification result to client
      res.json({
        ...classificationResult,
        imageUrl // Include the stored image URL
      });
    } catch (error) {
      console.error('Classification error:', error);
      res.status(500).json({ error: "Classification failed" });
    }
  });
}
