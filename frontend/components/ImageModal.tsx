"use client";

import React from "react";
import { useEffect } from "react";

interface ImageModalProps {
  isOpen: boolean;
  imageUrl: string | null;
  onClose: () => void;
}

export default function ImageModal({ isOpen, imageUrl, onClose }: ImageModalProps) {
  if (!isOpen || !imageUrl) return null;

  useEffect(() => {
    function handleEsc(event: KeyboardEvent) {
        if (event.key === "Escape") {
            onClose();
        }
    }

    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
     }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-lg bg-black/60"
      onClick={onClose}
    >
      <img
        src={imageUrl}
        className="max-w-[90%] max-h-[90%] rounded-xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
