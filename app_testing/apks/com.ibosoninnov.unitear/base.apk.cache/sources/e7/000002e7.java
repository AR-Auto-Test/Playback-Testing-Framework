package b.d.b.d1;

import b.d.b.a1;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Collection;

/* compiled from: CameraInternal.java */
/* loaded from: classes.dex */
public interface a0 extends b.d.b.e0, a1.b {

    /* compiled from: CameraInternal.java */
    /* loaded from: classes.dex */
    public enum a {
        PENDING_OPEN(false),
        OPENING(true),
        OPEN(true),
        CLOSING(true),
        CLOSED(false),
        RELEASING(true),
        RELEASED(false);
        
        public final boolean j;

        a(boolean z) {
            this.j = z;
        }
    }

    @Override // b.d.b.e0
    default b.d.b.f0 a() {
        return g();
    }

    @Override // b.d.b.e0
    default b.d.b.i0 b() {
        return j();
    }

    w g();

    void h(Collection<b.d.b.a1> collection);

    void i(Collection<b.d.b.a1> collection);

    z j();

    ListenableFuture<Void> release();
}