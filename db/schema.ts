import { pgTable, text, integer, timestamp, serial } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";

export const classifications = pgTable("classifications", {
  id: serial("id").primaryKey(),
  imageUrl: text("image_url").notNull(),
  breed: text("breed").notNull(),
  confidence: text("confidence").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertClassificationSchema = createInsertSchema(classifications);
export const selectClassificationSchema = createSelectSchema(classifications);
export type InsertClassification = z.infer<typeof insertClassificationSchema>;
export type Classification = z.infer<typeof selectClassificationSchema>;
