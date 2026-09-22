/** @type {import('next').NextConfig} */
const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle for the production Docker image.
  output: "standalone",
  poweredByHeader: false,
  async rewrites() {
    // The browser only talks to the frontend origin; /api/* is proxied to FastAPI.
    // This keeps auth cookies same-origin and avoids CORS in production.
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },
};

export default nextConfig;
