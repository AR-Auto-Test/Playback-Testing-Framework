package com.google.android.gms.common.api.internal;

import java.util.concurrent.locks.Lock;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public abstract class zaan implements Runnable {
    private final /* synthetic */ zaad zaa;

    private zaan(zaad zaadVar) {
        this.zaa = zaadVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        Lock lock;
        Lock lock2;
        zaax zaaxVar;
        lock = this.zaa.zab;
        lock.lock();
        try {
            if (Thread.interrupted()) {
                return;
            }
            zaa();
        } catch (RuntimeException e2) {
            zaaxVar = this.zaa.zaa;
            zaaxVar.zaa(e2);
        } finally {
            lock2 = this.zaa.zab;
            lock2.unlock();
        }
    }

    public abstract void zaa();

    public /* synthetic */ zaan(zaad zaadVar, zaag zaagVar) {
        this(zaadVar);
    }
}