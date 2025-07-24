package b.d.b.d1;

import android.util.Pair;
import android.util.Size;
import b.d.b.d1.i0;
import java.util.List;

/* compiled from: ImageOutputConfig.java */
/* loaded from: classes.dex */
public interface n0 extends a1 {

    /* renamed from: b  reason: collision with root package name */
    public static final i0.a<Integer> f1574b = new n("camerax.core.imageOutput.targetAspectRatio", b.d.b.b0.class, null);

    /* renamed from: c  reason: collision with root package name */
    public static final i0.a<Integer> f1575c = new n("camerax.core.imageOutput.targetRotation", Integer.TYPE, null);

    /* renamed from: d  reason: collision with root package name */
    public static final i0.a<Size> f1576d = new n("camerax.core.imageOutput.targetResolution", Size.class, null);

    /* renamed from: e  reason: collision with root package name */
    public static final i0.a<Size> f1577e = new n("camerax.core.imageOutput.defaultResolution", Size.class, null);

    /* renamed from: f  reason: collision with root package name */
    public static final i0.a<Size> f1578f = new n("camerax.core.imageOutput.maxResolution", Size.class, null);

    /* renamed from: g  reason: collision with root package name */
    public static final i0.a<List<Pair<Integer, Size[]>>> f1579g = new n("camerax.core.imageOutput.supportedResolutions", List.class, null);

    default Size i(Size size) {
        return (Size) f(f1578f, null);
    }

    default List<Pair<Integer, Size[]>> j(List<Pair<Integer, Size[]>> list) {
        return (List) f(f1579g, null);
    }

    default Size n(Size size) {
        return (Size) f(f1577e, null);
    }

    default Size o(Size size) {
        return (Size) f(f1576d, null);
    }

    default boolean q() {
        return b(f1574b);
    }

    default int s() {
        return ((Integer) a(f1574b)).intValue();
    }

    default int w(int i) {
        return ((Integer) f(f1575c, Integer.valueOf(i))).intValue();
    }
}