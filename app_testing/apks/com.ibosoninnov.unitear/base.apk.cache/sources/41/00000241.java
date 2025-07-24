package b.d.a.e;

import android.hardware.camera2.params.MeteringRectangle;
import b.d.a.e.o0;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;

/* compiled from: FocusMeteringControl.java */
/* loaded from: classes.dex */
public class l1 {

    /* renamed from: a  reason: collision with root package name */
    public final o0 f1092a;

    /* renamed from: b  reason: collision with root package name */
    public final Executor f1093b;

    /* renamed from: c  reason: collision with root package name */
    public final ScheduledExecutorService f1094c;

    /* renamed from: d  reason: collision with root package name */
    public volatile boolean f1095d = false;

    /* renamed from: e  reason: collision with root package name */
    public o0.c f1096e = null;

    /* renamed from: f  reason: collision with root package name */
    public MeteringRectangle[] f1097f = new MeteringRectangle[0];

    /* renamed from: g  reason: collision with root package name */
    public MeteringRectangle[] f1098g = new MeteringRectangle[0];

    /* renamed from: h  reason: collision with root package name */
    public MeteringRectangle[] f1099h = new MeteringRectangle[0];
    public MeteringRectangle[] i = new MeteringRectangle[0];
    public MeteringRectangle[] j = new MeteringRectangle[0];
    public MeteringRectangle[] k = new MeteringRectangle[0];
    public b.g.a.b<Void> l = null;

    public l1(o0 o0Var, ScheduledExecutorService scheduledExecutorService, Executor executor) {
        this.f1092a = o0Var;
        this.f1093b = executor;
        this.f1094c = scheduledExecutorService;
    }
}