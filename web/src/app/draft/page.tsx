"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { DraftRoom } from "@/components/DraftRoom";

function DraftPageInner() {
  const params = useSearchParams();
  const router = useRouter();
  const sessionId = params.get("session");
  return (
    <DraftRoom
      sessionId={sessionId}
      onSession={(id) => router.replace(`/draft?session=${id}`)}
    />
  );
}

export default function DraftPage() {
  return (
    <Suspense>
      <DraftPageInner />
    </Suspense>
  );
}
