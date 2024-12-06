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
      
      // Process the image using PyTorch service
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        body: req.file.buffer,
        headers: {
          'Content-Type': 'application/octet-stream'
        }
      });

      if (!response.ok) {
        throw new Error('Classification service failed');
      }

      const classificationResult = await response.json();
      const topPrediction = classificationResult.predictions[0];
      
      // Store classification result
      const result = await db.insert(classifications).values({
        imageUrl,
        breed: topPrediction.breed,
        confidence: topPrediction.confidence.toFixed(4),
      }).returning();

      res.json(result[0]);
    } catch (error) {
      console.error('Classification error:', error);
      res.status(500).json({ error: "Classification failed" });
    }
  });
}
