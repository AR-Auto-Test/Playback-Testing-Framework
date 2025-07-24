package b.d.b.d1;

import android.view.Surface;
import com.google.common.util.concurrent.ListenableFuture;

/* compiled from: ImmediateSurface.java */
/* loaded from: classes.dex */
public final class p0 extends j0 {
    public final Surface i;

    public p0(Surface surface) {
        this.i = surface;
    }

    @Override // b.d.b.d1.j0
    public ListenableFuture<Surface> g() {
        return b.d.b.d1.k1.c.g.c(this.i);
    }
}