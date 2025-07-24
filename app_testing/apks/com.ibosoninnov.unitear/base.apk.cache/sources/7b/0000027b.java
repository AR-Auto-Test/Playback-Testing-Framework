package b.d.a.e;

import android.content.Context;
import android.graphics.Point;
import android.hardware.camera2.CaptureRequest;
import android.util.ArrayMap;
import android.util.Size;
import android.view.WindowManager;
import b.d.b.d1.f0;
import b.d.b.d1.i0;
import b.d.b.d1.j1;
import java.util.ArrayList;
import java.util.HashSet;

/* compiled from: Camera2UseCaseConfigFactory.java */
/* loaded from: classes.dex */
public final class x0 implements b.d.b.d1.j1 {

    /* renamed from: a  reason: collision with root package name */
    public static final Size f1233a = new Size(1920, 1080);

    /* renamed from: b  reason: collision with root package name */
    public final WindowManager f1234b;

    public x0(Context context) {
        this.f1234b = (WindowManager) context.getSystemService("window");
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:0x00b0, code lost:
        if (r8 != 3) goto L14;
     */
    /* JADX WARN: Removed duplicated region for block: B:22:0x00d7 A[LOOP:0: B:20:0x00d1->B:22:0x00d7, LOOP_END] */
    /* JADX WARN: Removed duplicated region for block: B:25:0x00f9  */
    /* JADX WARN: Removed duplicated region for block: B:26:0x00fc  */
    /* JADX WARN: Removed duplicated region for block: B:29:0x0103  */
    /* JADX WARN: Removed duplicated region for block: B:37:0x0149  */
    @Override // b.d.b.d1.j1
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public b.d.b.d1.i0 a(j1.a aVar) {
        int i;
        x0 x0Var;
        Size size;
        i0.c cVar = i0.c.OPTIONAL;
        b.d.b.d1.u0 y = b.d.b.d1.u0.y();
        HashSet hashSet = new HashSet();
        f0.a aVar2 = new f0.a();
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList();
        ArrayList arrayList3 = new ArrayList();
        ArrayList arrayList4 = new ArrayList();
        aVar2.f1470c = 1;
        j1.a aVar3 = j1.a.PREVIEW;
        if (aVar == aVar3 && ((b.d.a.e.y1.p.i) b.d.a.e.y1.p.d.a(b.d.a.e.y1.p.i.class)) != null) {
            b.d.b.d1.u0 y2 = b.d.b.d1.u0.y();
            CaptureRequest.Key key = CaptureRequest.TONEMAP_MODE;
            i0.a<Integer> aVar4 = b.d.a.d.a.r;
            StringBuilder x = c.b.a.a.a.x("camera2.captureRequest.option.");
            x.append(key.getName());
            y2.A(new b.d.b.d1.n(x.toString(), Object.class, key), cVar, 2);
            aVar2.c(new b.d.a.d.a(b.d.b.d1.w0.x(y2)));
        }
        y.A(b.d.b.d1.i1.f1495h, cVar, new b.d.b.d1.b1(new ArrayList(hashSet), arrayList, arrayList2, arrayList4, arrayList3, aVar2.d()));
        y.A(b.d.b.d1.i1.j, cVar, w0.f1223a);
        HashSet hashSet2 = new HashSet();
        b.d.b.d1.u0 y3 = b.d.b.d1.u0.y();
        int i2 = -1;
        ArrayList arrayList5 = new ArrayList();
        ArrayMap arrayMap = new ArrayMap();
        b.d.b.d1.v0 v0Var = new b.d.b.d1.v0(arrayMap);
        int ordinal = aVar.ordinal();
        if (ordinal != 0) {
            i = 1;
            if (ordinal != 1) {
                if (ordinal != 2) {
                }
            }
            i0.a<b.d.b.d1.f0> aVar5 = b.d.b.d1.i1.i;
            ArrayList arrayList6 = new ArrayList(hashSet2);
            b.d.b.d1.w0 x2 = b.d.b.d1.w0.x(y3);
            b.d.b.d1.g1 g1Var = b.d.b.d1.g1.f1479a;
            ArrayMap arrayMap2 = new ArrayMap();
            for (String str : arrayMap.keySet()) {
                arrayMap2.put(str, v0Var.a(str));
            }
            y.A(aVar5, cVar, new b.d.b.d1.f0(arrayList6, x2, i, arrayList5, false, new b.d.b.d1.g1(arrayMap2)));
            y.A(b.d.b.d1.i1.k, cVar, aVar != j1.a.IMAGE_CAPTURE ? m1.f1101b : u0.f1205a);
            if (aVar != aVar3) {
                i0.a<Size> aVar6 = b.d.b.d1.n0.f1578f;
                Point point = new Point();
                x0Var = this;
                x0Var.f1234b.getDefaultDisplay().getRealSize(point);
                if (point.x > point.y) {
                    size = new Size(point.x, point.y);
                } else {
                    size = new Size(point.y, point.x);
                }
                int height = size.getHeight() * size.getWidth();
                Size size2 = f1233a;
                if (height > size2.getHeight() * size2.getWidth()) {
                    size = size2;
                }
                y.A(aVar6, cVar, size);
            } else {
                x0Var = this;
            }
            y.A(b.d.b.d1.n0.f1575c, cVar, Integer.valueOf(x0Var.f1234b.getDefaultDisplay().getRotation()));
            return b.d.b.d1.w0.x(y);
        }
        i2 = 2;
        i = i2;
        i0.a<b.d.b.d1.f0> aVar52 = b.d.b.d1.i1.i;
        ArrayList arrayList62 = new ArrayList(hashSet2);
        b.d.b.d1.w0 x22 = b.d.b.d1.w0.x(y3);
        b.d.b.d1.g1 g1Var2 = b.d.b.d1.g1.f1479a;
        ArrayMap arrayMap22 = new ArrayMap();
        while (r7.hasNext()) {
        }
        y.A(aVar52, cVar, new b.d.b.d1.f0(arrayList62, x22, i, arrayList5, false, new b.d.b.d1.g1(arrayMap22)));
        y.A(b.d.b.d1.i1.k, cVar, aVar != j1.a.IMAGE_CAPTURE ? m1.f1101b : u0.f1205a);
        if (aVar != aVar3) {
        }
        y.A(b.d.b.d1.n0.f1575c, cVar, Integer.valueOf(x0Var.f1234b.getDefaultDisplay().getRotation()));
        return b.d.b.d1.w0.x(y);
    }
}