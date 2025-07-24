package b.d.a.e;

import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.util.Size;
import android.view.Surface;
import b.d.b.d1.b1;
import b.d.b.d1.i0;
import b.d.b.d1.k1.c.g;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Arrays;
import java.util.Collections;

/* compiled from: MeteringRepeatingSession.java */
/* loaded from: classes.dex */
public class n1 {

    /* renamed from: a  reason: collision with root package name */
    public b.d.b.d1.j0 f1105a;

    /* renamed from: b  reason: collision with root package name */
    public final b.d.b.d1.b1 f1106b;

    /* compiled from: MeteringRepeatingSession.java */
    /* loaded from: classes.dex */
    public class a implements b.d.b.d1.k1.c.d<Void> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ Surface f1107a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ SurfaceTexture f1108b;

        public a(n1 n1Var, Surface surface, SurfaceTexture surfaceTexture) {
            this.f1107a = surface;
            this.f1108b = surfaceTexture;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            throw new IllegalStateException("Future should never fail. Did it get completed by GC?", th);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r1) {
            this.f1107a.release();
            this.f1108b.release();
        }
    }

    /* compiled from: MeteringRepeatingSession.java */
    /* loaded from: classes.dex */
    public static class b implements b.d.b.d1.i1<b.d.b.a1> {
        public final b.d.b.d1.i0 q;

        public b() {
            b.d.b.d1.u0 y = b.d.b.d1.u0.y();
            y.A(b.d.b.d1.i1.j, i0.c.OPTIONAL, new w0());
            this.q = y;
        }

        @Override // b.d.b.d1.a1
        public b.d.b.d1.i0 k() {
            return this.q;
        }
    }

    public n1(b.d.a.e.y1.e eVar) {
        Size size;
        b bVar = new b();
        SurfaceTexture surfaceTexture = new SurfaceTexture(0);
        StreamConfigurationMap streamConfigurationMap = (StreamConfigurationMap) eVar.a(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (streamConfigurationMap == null) {
            b.d.b.u0.b("MeteringRepeating", "Can not retrieve SCALER_STREAM_CONFIGURATION_MAP.", null);
            size = new Size(0, 0);
        } else {
            Size[] outputSizes = streamConfigurationMap.getOutputSizes(34);
            if (outputSizes == null) {
                b.d.b.u0.b("MeteringRepeating", "Can not get output size list.", null);
                size = new Size(0, 0);
            } else {
                size = (Size) Collections.min(Arrays.asList(outputSizes), z.f1355b);
            }
        }
        b.d.b.u0.a("MeteringRepeating", "MerteringSession SurfaceTexture size: " + size, null);
        surfaceTexture.setDefaultBufferSize(size.getWidth(), size.getHeight());
        Surface surface = new Surface(surfaceTexture);
        b1.b e2 = b1.b.e(bVar);
        e2.f1421b.f1470c = 1;
        b.d.b.d1.p0 p0Var = new b.d.b.d1.p0(surface);
        this.f1105a = p0Var;
        ListenableFuture<Void> d2 = p0Var.d();
        a aVar = new a(this, surface, surfaceTexture);
        d2.addListener(new g.d(d2, aVar), b.b.a.f());
        b.d.b.d1.j0 j0Var = this.f1105a;
        e2.f1420a.add(j0Var);
        e2.f1421b.f1468a.add(j0Var);
        this.f1106b = e2.d();
    }
}