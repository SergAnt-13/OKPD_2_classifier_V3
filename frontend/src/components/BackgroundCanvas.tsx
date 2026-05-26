import { useEffect, useRef } from "react";

export function BackgroundCanvas(): JSX.Element {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let frame = 0;
    let animation = 0;
    const particles = Array.from({ length: 55 }, () => ({
      x: Math.random(),
      y: Math.random(),
      size: Math.random() * 2 + 0.5,
      speed: Math.random() * 0.0007 + 0.0002
    }));

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const draw = () => {
      frame += 1;
      context.clearRect(0, 0, canvas.width, canvas.height);
      const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
      gradient.addColorStop(0, "rgba(0, 240, 255, 0.08)");
      gradient.addColorStop(0.5, "rgba(15, 23, 42, 0.02)");
      gradient.addColorStop(1, "rgba(168, 85, 247, 0.1)");
      context.fillStyle = gradient;
      context.fillRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, index) => {
        particle.y += particle.speed;
        if (particle.y > 1.1) particle.y = -0.1;
        const x = particle.x * canvas.width + Math.sin(frame * 0.01 + index) * 14;
        const y = particle.y * canvas.height;
        context.beginPath();
        context.fillStyle = index % 2 === 0 ? "rgba(0,240,255,0.36)" : "rgba(168,85,247,0.28)";
        context.arc(x, y, particle.size, 0, Math.PI * 2);
        context.fill();
      });

      animation = window.requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener("resize", resize);
    return () => {
      window.cancelAnimationFrame(animation);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={ref} className="pointer-events-none fixed inset-0 -z-10 opacity-90" />;
}
