import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card } from "@/components/ui/card";
import { PawPrint, Upload } from "lucide-react";

interface UploadZoneProps {
  onUpload: (file: File) => void;
}

export default function UploadZone({ onUpload }: UploadZoneProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles[0]) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1
  });

  return (
    <Card
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-8
        flex flex-col items-center justify-center
        cursor-pointer transition-colors
        ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300'}
      `}
    >
      <input {...getInputProps()} />
      
      <PawPrint className="w-12 h-12 text-primary mb-4" />
      
      <div className="text-center">
        <p className="text-lg font-medium">
          {isDragActive
            ? "Drop your dog's photo here"
            : "Drag & drop your dog's photo here"}
        </p>
        <p className="text-sm text-gray-500 mt-2">
          or click to select a file
        </p>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Upload className="w-4 h-4" />
        <span className="text-sm text-gray-500">
          Supports JPG, JPEG, PNG
        </span>
      </div>
    </Card>
  );
}
