"use client";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { ConnectYahooForm } from "@/components/ConnectYahooForm";
import { DraftRoom } from "@/components/DraftRoom";

function DraftPageInner() {
  const params = useSearchParams();
  const sessionId = params.get("session");
  const connected = params.get("connected") === "1";
  const connectError = params.get("connect_error");

  if (!sessionId) {
    // connect_error can land here with no session (e.g. an unparseable OAuth
    // state) — show the generic error above the form rather than silently
    // dropping it, since DraftRoom's error branch only handles the case
    // where a session_id is present.
    return (
      <div className="p-8">
        <h1 className="text-xl font-bold mb-4">Connect your Yahoo league</h1>
        {connectError && (
          <p className="text-red-700 text-sm mb-4">
            Something went wrong connecting to Yahoo. Please try again.
          </p>
        )}
        <ConnectYahooForm />
      </div>
    );
  }
  return <DraftRoom sessionId={sessionId} connected={connected} connectError={connectError} />;
}

export default function DraftPage() {
  return (
    <Suspense>
      <DraftPageInner />
    </Suspense>
  );
}
