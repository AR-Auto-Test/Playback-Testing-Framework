package com.google.ar.core;

import android.os.Bundle;
import android.util.Log;
import com.google.ar.core.exceptions.FatalException;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class aa extends com.google.ar.core.dependencies.j {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ AtomicBoolean f5540a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ac f5541b;

    public aa(ac acVar, AtomicBoolean atomicBoolean) {
        this.f5541b = acVar;
        this.f5540a = atomicBoolean;
    }

    @Override // com.google.ar.core.dependencies.k
    public final void b(Bundle bundle) {
    }

    @Override // com.google.ar.core.dependencies.k
    public final void c(Bundle bundle) {
        if (this.f5540a.getAndSet(true)) {
            return;
        }
        int i = bundle.getInt("error.code", -100);
        int i2 = bundle.getInt("install.status", 0);
        if (i2 == 4) {
            this.f5541b.f5545b.a(t.COMPLETED);
        } else if (i != 0) {
            StringBuilder sb = new StringBuilder(51);
            sb.append("requestInstall = ");
            sb.append(i);
            sb.append(", launching fullscreen.");
            Log.w("ARCore-InstallService", sb.toString());
            ac acVar = this.f5541b;
            u uVar = acVar.f5546c;
            u.o(acVar.f5544a, acVar.f5545b);
        } else if (bundle.containsKey("resolution.intent")) {
            ac acVar2 = this.f5541b;
            u uVar2 = acVar2.f5546c;
            u.p(acVar2.f5544a, bundle, acVar2.f5545b);
        } else if (i2 != 10) {
            switch (i2) {
                case 1:
                case 2:
                case 3:
                    this.f5541b.f5545b.a(t.ACCEPTED);
                    return;
                case 4:
                    this.f5541b.f5545b.a(t.COMPLETED);
                    return;
                case 5:
                    this.f5541b.f5545b.b(new FatalException("Unexpected FAILED install status without error."));
                    return;
                case 6:
                    this.f5541b.f5545b.a(t.CANCELLED);
                    return;
                default:
                    this.f5541b.f5545b.b(new FatalException(c.b.a.a.a.g(38, "Unexpected install status: ", i2)));
                    return;
            }
        } else {
            this.f5541b.f5545b.b(new FatalException("Unexpected REQUIRES_UI_INTENT install status without an intent."));
        }
    }
}