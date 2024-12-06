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
  const { data, isLoading } = useQuery({
    queryKey: ["classify", image],
    queryFn: () => classifyImage(image),
    enabled: !processing,
  });

  if (processing || isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-4 w-[200px]" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
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
    </div>
  );
}
