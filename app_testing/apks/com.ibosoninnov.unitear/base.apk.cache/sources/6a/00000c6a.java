package com.google.android.datatransport.runtime.scheduling.jobscheduling;

import com.google.android.datatransport.runtime.TransportContext;
import com.google.android.datatransport.runtime.scheduling.jobscheduling.WorkInitializer;
import com.google.android.datatransport.runtime.scheduling.persistence.EventStore;
import com.google.android.datatransport.runtime.synchronization.SynchronizationGuard;
import java.util.concurrent.Executor;

/* loaded from: classes.dex */
public class WorkInitializer {
    private final Executor executor;
    private final SynchronizationGuard guard;
    private final WorkScheduler scheduler;
    private final EventStore store;

    public WorkInitializer(Executor executor, EventStore eventStore, WorkScheduler workScheduler, SynchronizationGuard synchronizationGuard) {
        this.executor = executor;
        this.store = eventStore;
        this.scheduler = workScheduler;
        this.guard = synchronizationGuard;
    }

    public /* synthetic */ Object a() {
        for (TransportContext transportContext : this.store.loadActiveContexts()) {
            this.scheduler.schedule(transportContext, 1);
        }
        return null;
    }

    public /* synthetic */ void b() {
        this.guard.runCriticalSection(new SynchronizationGuard.CriticalSection() { // from class: c.d.a.a.b.b.c.m
            @Override // com.google.android.datatransport.runtime.synchronization.SynchronizationGuard.CriticalSection
            public final Object execute() {
                WorkInitializer.this.a();
                return null;
            }
        });
    }

    public void ensureContextsScheduled() {
        this.executor.execute(new Runnable() { // from class: c.d.a.a.b.b.c.n
            @Override // java.lang.Runnable
            public final void run() {
                WorkInitializer.this.b();
            }
        });
    }
}