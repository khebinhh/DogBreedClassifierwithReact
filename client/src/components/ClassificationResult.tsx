import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { classifyImage, type ClassificationResult, isValidClassificationResult } from "../lib/ml";
import { useState, useEffect } from "react";

interface ClassificationResultProps {
  image: File;
  processing: boolean;
}

interface ValidationError {
  message: string;
}

export default function ClassificationResult({ image, processing }: ClassificationResultProps) {
  const [validationError, setValidationError] = useState<ValidationError | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["classify", image],
    queryFn: () => classifyImage(image),
    enabled: !processing,
    retry: 1,
  });

  useEffect(() => {
    if (data && !isValidClassificationResult(data)) {
      setValidationError({ message: "Invalid response format from server" });
    } else {
      setValidationError(null);
    }
  }, [data]);

  // Type guard assertion for TypeScript
  const result: ClassificationResult | undefined = 
    data && isValidClassificationResult(data) ? data : undefined;

  // Create object URL for image preview
  const imageUrl = image ? URL.createObjectURL(image) : null;

  if (error || validationError) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50">
        <p className="text-red-600">
          {validationError ? validationError.message : "Failed to classify image. Please try again."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {imageUrl && (
        <div className="aspect-square w-full max-w-md mx-auto overflow-hidden rounded-lg">
          <img
            src={imageUrl}
            alt="Uploaded"
            className="w-full h-full object-cover"
            onLoad={() => URL.revokeObjectURL(imageUrl)}
          />
        </div>
      )}

      {(processing || isLoading) ? (
        <div className="space-y-4">
          <Skeleton className="h-4 w-[200px]" />
          <Skeleton className="h-32 w-full" />
          <div className="grid grid-cols-3 gap-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        </div>
      ) : result ? (
        <>
          <h3 className="text-xl font-semibold">Classification Results</h3>
          
          <div className="grid gap-4">
            {result.predictions.map((pred, idx) => (
              <Card key={idx} className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">{pred.breed}</span>
                  <span className="text-sm text-gray-500">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={pred.confidence * 100} />
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {result.referenceImages.map((img, idx) => (
              <Card key={idx} className="p-2 overflow-hidden">
                <div className="aspect-square relative">
                  <img
                    src={img}
                    alt={`Reference ${result.predictions[idx].breed}`}
                    className="absolute inset-0 w-full h-full object-cover rounded-lg"
                    onError={(e) => {
                      const breed = result.predictions[idx].breed.replace(' ', '+');
                      e.currentTarget.src = `https://dog.ceo/api/breed/${breed.toLowerCase()}/images/random`;
                    }}
                  />
                </div>
                <p className="mt-2 text-sm text-center text-gray-600">
                  {result.predictions[idx].breed}
                </p>
              </Card>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
