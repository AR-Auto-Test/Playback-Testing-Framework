package b.d.b.d1;

import b.d.b.d1.a0;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executor;

/* compiled from: CameraStateRegistry.java */
/* loaded from: classes.dex */
public final class c0 {

    /* renamed from: c  reason: collision with root package name */
    public final int f1435c;

    /* renamed from: e  reason: collision with root package name */
    public int f1437e;

    /* renamed from: a  reason: collision with root package name */
    public final StringBuilder f1433a = new StringBuilder();

    /* renamed from: b  reason: collision with root package name */
    public final Object f1434b = new Object();

    /* renamed from: d  reason: collision with root package name */
    public final Map<b.d.b.e0, a> f1436d = new HashMap();

    /* compiled from: CameraStateRegistry.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public a0.a f1438a = null;

        /* renamed from: b  reason: collision with root package name */
        public final Executor f1439b;

        /* renamed from: c  reason: collision with root package name */
        public final b f1440c;

        public a(a0.a aVar, Executor executor, b bVar) {
            this.f1439b = executor;
            this.f1440c = bVar;
        }
    }

    /* compiled from: CameraStateRegistry.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public c0(int i) {
        this.f1435c = i;
        synchronized ("mLock") {
            this.f1437e = i;
        }
    }

    public static boolean a(a0.a aVar) {
        return aVar != null && aVar.j;
    }

    public final void b() {
        if (b.d.b.u0.c("CameraStateRegistry")) {
            this.f1433a.setLength(0);
            this.f1433a.append("Recalculating open cameras:\n");
            this.f1433a.append(String.format(Locale.US, "%-45s%-22s\n", "Camera", "State"));
            this.f1433a.append("-------------------------------------------------------------------\n");
        }
        int i = 0;
        for (Map.Entry<b.d.b.e0, a> entry : this.f1436d.entrySet()) {
            if (b.d.b.u0.c("CameraStateRegistry")) {
                this.f1433a.append(String.format(Locale.US, "%-45s%-22s\n", entry.getKey().toString(), entry.getValue().f1438a != null ? entry.getValue().f1438a.toString() : "UNKNOWN"));
            }
            if (a(entry.getValue().f1438a)) {
                i++;
            }
        }
        if (b.d.b.u0.c("CameraStateRegistry")) {
            this.f1433a.append("-------------------------------------------------------------------\n");
            this.f1433a.append(String.format(Locale.US, "Open count: %d (Max allowed: %d)", Integer.valueOf(i), Integer.valueOf(this.f1435c)));
            b.d.b.u0.a("CameraStateRegistry", this.f1433a.toString(), null);
        }
        this.f1437e = Math.max(this.f1435c - i, 0);
    }
}