package b.d.a.e;

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.TotalCaptureResult;
import android.os.Build;
import b.d.a.d.a;
import b.d.a.e.o0;
import java.util.concurrent.Executor;

/* compiled from: ZoomControl.java */
/* loaded from: classes.dex */
public final class w1 {

    /* renamed from: a  reason: collision with root package name */
    public final o0 f1224a;

    /* renamed from: b  reason: collision with root package name */
    public final Executor f1225b;

    /* renamed from: c  reason: collision with root package name */
    public final x1 f1226c;

    /* renamed from: d  reason: collision with root package name */
    public final b.t.m<b.d.b.c1> f1227d;

    /* renamed from: e  reason: collision with root package name */
    public final b f1228e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f1229f = false;

    /* renamed from: g  reason: collision with root package name */
    public o0.c f1230g = new a();

    /* compiled from: ZoomControl.java */
    /* loaded from: classes.dex */
    public class a implements o0.c {
        public a() {
        }

        @Override // b.d.a.e.o0.c
        public boolean a(TotalCaptureResult totalCaptureResult) {
            w1.this.f1228e.a(totalCaptureResult);
            return false;
        }
    }

    /* compiled from: ZoomControl.java */
    /* loaded from: classes.dex */
    public interface b {
        void a(TotalCaptureResult totalCaptureResult);

        void b(a.C0012a c0012a);

        float c();

        float d();

        void e();
    }

    public w1(o0 o0Var, b.d.a.e.y1.e eVar, Executor executor) {
        b i1Var;
        boolean z = false;
        this.f1224a = o0Var;
        this.f1225b = executor;
        if (Build.VERSION.SDK_INT >= 30 && eVar.a(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) != null) {
            z = true;
        }
        if (z) {
            i1Var = new l0(eVar);
        } else {
            i1Var = new i1(eVar);
        }
        this.f1228e = i1Var;
        x1 x1Var = new x1(i1Var.c(), i1Var.d());
        this.f1226c = x1Var;
        x1Var.a(1.0f);
        this.f1227d = new b.t.m<>(b.d.b.e1.d.a(x1Var));
        o0Var.e(this.f1230g);
    }
}