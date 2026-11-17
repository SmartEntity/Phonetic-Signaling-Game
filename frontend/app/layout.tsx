import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "One round",
  description: "Walk through a single round of the signaling game: the object, the word, the guess.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Charis SIL is a typeface made for phonetics, so every IPA symbol the speaker can produce has a glyph. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Charis+SIL:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
