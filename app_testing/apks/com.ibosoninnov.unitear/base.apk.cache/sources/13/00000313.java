package b.d.b.d1;

import b.d.b.a1;
import b.d.b.d1.b1;
import b.d.b.d1.f0;
import b.d.b.d1.i0;

/* compiled from: UseCaseConfig.java */
/* loaded from: classes.dex */
public interface i1<T extends b.d.b.a1> extends b.d.b.e1.e<T>, b.d.b.e1.g, m0 {

    /* renamed from: h  reason: collision with root package name */
    public static final i0.a<b1> f1495h = new n("camerax.core.useCase.defaultSessionConfig", b1.class, null);
    public static final i0.a<f0> i = new n("camerax.core.useCase.defaultCaptureConfig", f0.class, null);
    public static final i0.a<b1.d> j = new n("camerax.core.useCase.sessionConfigUnpacker", b1.d.class, null);
    public static final i0.a<f0.b> k = new n("camerax.core.useCase.captureConfigUnpacker", f0.b.class, null);
    public static final i0.a<Integer> l = new n("camerax.core.useCase.surfaceOccupancyPriority", Integer.TYPE, null);
    public static final i0.a<b.d.b.j0> m = new n("camerax.core.useCase.cameraSelector", b.d.b.j0.class, null);

    /* compiled from: UseCaseConfig.java */
    /* loaded from: classes.dex */
    public interface a<T extends b.d.b.a1, C extends i1<T>, B> {
    }

    /* JADX DEBUG: Type inference failed for r2v1. Raw type applied. Possible types: b.d.b.d1.i0$a<b.d.b.d1.b1>, b.d.b.d1.i0$a<ValueT> */
    default b1 m(b1 b1Var) {
        return (b1) f(f1495h, null);
    }

    /* JADX DEBUG: Type inference failed for r0v0. Raw type applied. Possible types: b.d.b.d1.i0$a<java.lang.Integer>, b.d.b.d1.i0$a<ValueT> */
    default int r(int i2) {
        return ((Integer) f(l, Integer.valueOf(i2))).intValue();
    }

    /* JADX DEBUG: Type inference failed for r2v1. Raw type applied. Possible types: b.d.b.d1.i0$a<b.d.b.j0>, b.d.b.d1.i0$a<ValueT> */
    default b.d.b.j0 t(b.d.b.j0 j0Var) {
        return (b.d.b.j0) f(m, null);
    }

    /* JADX DEBUG: Type inference failed for r2v1. Raw type applied. Possible types: b.d.b.d1.i0$a<b.d.b.d1.b1$d>, b.d.b.d1.i0$a<ValueT> */
    default b1.d v(b1.d dVar) {
        return (b1.d) f(j, null);
    }
}