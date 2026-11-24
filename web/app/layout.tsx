import type { Metadata } from "next";
import "@fontsource-variable/manrope";
import "./globals.css";

export const metadata: Metadata = {
  title: "CV Analyzer | Analisis kecocokan CV",
  description: "Bandingkan CV dengan posisi tujuan dan susun rencana pengembangan skill.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
