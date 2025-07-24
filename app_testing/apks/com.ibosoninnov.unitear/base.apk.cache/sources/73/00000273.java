package b.d.a.e;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import b.d.a.e.o0;
import java.util.concurrent.Executor;

/* compiled from: TorchControl.java */
/* loaded from: classes.dex */
public final class v1 {

    /* renamed from: a  reason: collision with root package name */
    public final o0 f1211a;

    /* renamed from: b  reason: collision with root package name */
    public final b.t.m<Integer> f1212b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f1213c;

    /* renamed from: d  reason: collision with root package name */
    public final Executor f1214d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f1215e;

    /* renamed from: f  reason: collision with root package name */
    public b.g.a.b<Void> f1216f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f1217g;

    /* renamed from: h  reason: collision with root package name */
    public final o0.c f1218h;

    /* compiled from: TorchControl.java */
    /* loaded from: classes.dex */
    public class a implements o0.c {
        public a() {
        }

        @Override // b.d.a.e.o0.c
        public boolean a(TotalCaptureResult totalCaptureResult) {
            if (v1.this.f1216f != null) {
                Integer num = (Integer) totalCaptureResult.getRequest().get(CaptureRequest.FLASH_MODE);
                boolean z = num != null && num.intValue() == 2;
                v1 v1Var = v1.this;
                if (z == v1Var.f1217g) {
                    v1Var.f1216f.a(null);
                    v1.this.f1216f = null;
                }
            }
            return false;
        }
    }

    public v1(o0 o0Var, b.d.a.e.y1.e eVar, Executor executor) {
        a aVar = new a();
        this.f1218h = aVar;
        this.f1211a = o0Var;
        this.f1214d = executor;
        Boolean bool = (Boolean) eVar.a(CameraCharacteristics.FLASH_INFO_AVAILABLE);
        this.f1213c = bool != null && bool.booleanValue();
        this.f1212b = new b.t.m<>(0);
        o0Var.f1111b.f1120a.add(aVar);
    }

    public final <T> void a(b.t.m<T> mVar, T t) {
        if (b.b.a.k()) {
            mVar.h(t);
        } else {
            mVar.i(t);
        }
    }
}