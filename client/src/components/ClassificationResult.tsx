import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { classifyImage } from "../lib/ml";

interface ClassificationResultProps {
  image: File;
  processing: boolean;
}

export default function ClassificationResult({ image, processing }: ClassificationResultProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["classify", image],
    queryFn: () => classifyImage(image),
    enabled: !processing,
    retry: 1,
  });

  // Create object URL for image preview
  const imageUrl = image ? URL.createObjectURL(image) : null;

  if (error) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50">
        <p className="text-red-600">Failed to classify image. Please try again.</p>
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
      ) : (
        <>
          <h3 className="text-xl font-semibold">Classification Results</h3>
          
          <div className="grid gap-4">
            {data?.predictions.map((pred, idx) => (
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
            {data?.referenceImages.map((img, idx) => (
              <img
                key={idx}
                src={img}
                alt={`Reference ${idx + 1}`}
                className="rounded-lg object-cover aspect-square"
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
