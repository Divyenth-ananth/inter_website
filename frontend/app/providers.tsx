"use client";

import { ThemeProvider } from "@/components/theme-provider";

export default function ClientProviders({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ThemeProvider>{children}</ThemeProvider>;
}
