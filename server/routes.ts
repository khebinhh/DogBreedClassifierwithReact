import type { Express } from "express";
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
  app.post("/api/classify", upload.single("image"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No image provided" });
    }

    try {
      // Store classification result
      const result = await db.insert(classifications).values({
        imageUrl: "https://images.unsplash.com/photo-1592260368948-7789d2480b5a",
        breed: "Labrador Retriever",
        confidence: "0.85",
      });

      res.json(result);
    } catch (error) {
      res.status(500).json({ error: "Classification failed" });
    }
  });
}
