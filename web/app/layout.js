import { Space_Grotesk, Sora } from "next/font/google";
import "./globals.css";

const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display"
});

const sans = Sora({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans"
});

export const metadata = {
  title: "ET Reel Studio",
  description: "Generate news reels from trusted articles."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable}`}>
      <body>{children}</body>
    </html>
  );
}
