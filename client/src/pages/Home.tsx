import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import UploadZone from "../components/UploadZone";
import ClassificationResult from "../components/ClassificationResult";
import HistoryGallery from "../components/HistoryGallery";
import { useToast } from "@/hooks/use-toast";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const { toast } = useToast();

  const handleImageUpload = async (file: File) => {
    setSelectedImage(file);
    setProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(errorData || "Classification failed");
      }

      const result = await response.json();
      if (!result.predictions || !Array.isArray(result.predictions)) {
        throw new Error("Invalid response format");
      }
      
    } catch (error: any) {
      console.error("Classification error:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to process image. Please try again.",
        variant: "destructive",
      });
      setSelectedImage(null);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col items-center gap-8">
        <h1 className="text-4xl font-bold text-primary">Dog Breed Classifier</h1>
        
        <Card className="w-full max-w-3xl p-6">
          <UploadZone onUpload={handleImageUpload} />
          
          {selectedImage && (
            <>
              <Separator className="my-6" />
              <ClassificationResult 
                image={selectedImage}
                processing={processing}
              />
            </>
          )}
        </Card>

        <Separator className="my-6 w-full max-w-3xl" />
        
        <Card className="w-full max-w-3xl p-6">
          <h2 className="text-2xl font-semibold mb-4">Recent Classifications</h2>
          <HistoryGallery />
        </Card>
      </div>
    </div>
  );
}
