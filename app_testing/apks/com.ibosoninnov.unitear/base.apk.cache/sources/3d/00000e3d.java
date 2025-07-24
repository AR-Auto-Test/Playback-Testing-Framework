package com.google.android.gms.common.api.internal;

import java.util.concurrent.locks.Lock;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zat implements Runnable {
    private final /* synthetic */ zaq zaa;

    public zat(zaq zaqVar) {
        this.zaa = zaqVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        Lock lock;
        Lock lock2;
        lock = this.zaa.zam;
        lock.lock();
        try {
            this.zaa.zah();
        } finally {
            lock2 = this.zaa.zam;
            lock2.unlock();
        }
    }
}