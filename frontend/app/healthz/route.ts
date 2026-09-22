// Liveness probe for Docker/Kubernetes. Deliberately independent of the backend.
export const dynamic = "force-dynamic";

export function GET() {
  return Response.json({ status: "ok" });
}
