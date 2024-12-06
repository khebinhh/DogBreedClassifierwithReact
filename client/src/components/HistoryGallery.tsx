import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function HistoryGallery() {
  const { data, isLoading } = useQuery({
    queryKey: ["classifications"],
    queryFn: async () => {
      const response = await fetch("/api/classifications");
      if (!response.ok) throw new Error("Failed to fetch history");
      return response.json();
    },
  });

  if (isLoading) {
    return <div>Loading history...</div>;
  }

  return (
    <ScrollArea className="h-[400px]">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {data?.map((item: any) => (
          <Card key={item.id} className="p-2">
            <img
              src={item.imageUrl}
              alt={item.breed}
              className="rounded-lg object-cover aspect-square mb-2"
            />
            <div className="p-2">
              <p className="font-medium">{item.breed}</p>
              <p className="text-sm text-gray-500">
                {new Date(item.createdAt).toLocaleDateString()}
              </p>
            </div>
          </Card>
        ))}
      </div>
    </ScrollArea>
  );
}
