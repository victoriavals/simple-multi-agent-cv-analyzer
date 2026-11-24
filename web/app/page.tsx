import Image from "next/image";
import Workspace from "@/components/workspace";

export default function Home() {
  return (
    <Workspace heroArt={
      <Image
        src="/cv-inspection.webp"
        alt=""
        width={840}
        height={442}
        className="hero-art"
        sizes="(max-width: 560px) 155px, (max-width: 840px) 260px, 440px"
        preload
      />
    } />
  );
}
