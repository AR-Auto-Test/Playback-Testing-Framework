package b.d.a.e;

import android.content.Context;
import android.graphics.Point;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaRecorder;
import android.util.Rational;
import android.util.Size;
import android.view.WindowManager;
import b.d.b.d1.e1;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/* compiled from: SupportedSurfaceCombination.java */
/* loaded from: classes.dex */
public final class o1 {

    /* renamed from: a  reason: collision with root package name */
    public static final Size f1122a = new Size(1920, 1080);

    /* renamed from: b  reason: collision with root package name */
    public static final Size f1123b = new Size(640, 480);

    /* renamed from: c  reason: collision with root package name */
    public static final Size f1124c = new Size(0, 0);

    /* renamed from: d  reason: collision with root package name */
    public static final Size f1125d = new Size(3840, 2160);

    /* renamed from: e  reason: collision with root package name */
    public static final Size f1126e = new Size(1920, 1080);

    /* renamed from: f  reason: collision with root package name */
    public static final Size f1127f = new Size(1280, 720);

    /* renamed from: g  reason: collision with root package name */
    public static final Size f1128g = new Size(720, 480);

    /* renamed from: h  reason: collision with root package name */
    public static final Rational f1129h = new Rational(4, 3);
    public static final Rational i = new Rational(3, 4);
    public static final Rational j = new Rational(16, 9);
    public static final Rational k = new Rational(9, 16);
    public final List<b.d.b.d1.d1> l;
    public final Map<Integer, Size> m;
    public final String n;
    public final m0 o;
    public final b.d.a.e.y1.e p;
    public final b.d.a.e.y1.q.c q;
    public final int r;
    public final boolean s;
    public final Map<Integer, List<Size>> t;
    public boolean u;
    public boolean v;
    public b.d.b.d1.f1 w;
    public Map<Integer, Size[]> x;

    /* compiled from: SupportedSurfaceCombination.java */
    /* loaded from: classes.dex */
    public static final class a implements Comparator<Rational> {

        /* renamed from: b  reason: collision with root package name */
        public Rational f1130b;

        public a(Rational rational) {
            this.f1130b = rational;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(Rational rational, Rational rational2) {
            Rational rational3 = rational;
            Rational rational4 = rational2;
            if (rational3.equals(rational4)) {
                return 0;
            }
            return (int) Math.signum(Float.valueOf(Math.abs(rational3.floatValue() - this.f1130b.floatValue())).floatValue() - Float.valueOf(Math.abs(rational4.floatValue() - this.f1130b.floatValue())).floatValue());
        }
    }

    public o1(Context context, String str, b.d.a.e.y1.k kVar, m0 m0Var) {
        WindowManager windowManager;
        Size size;
        ArrayList arrayList = new ArrayList();
        this.l = arrayList;
        this.m = new HashMap();
        this.t = new HashMap();
        boolean z = false;
        this.u = false;
        this.v = false;
        this.x = new HashMap();
        Objects.requireNonNull(str);
        this.n = str;
        Objects.requireNonNull(m0Var);
        this.o = m0Var;
        WindowManager windowManager2 = (WindowManager) context.getSystemService("window");
        this.q = new b.d.a.e.y1.q.c(str);
        try {
            b.d.a.e.y1.e b2 = kVar.b(str);
            this.p = b2;
            Integer num = (Integer) b2.a(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
            int intValue = num != null ? num.intValue() : 2;
            this.r = intValue;
            Size size2 = (Size) b2.a(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
            this.s = (size2 == null || size2.getWidth() >= size2.getHeight()) ? true : true;
            e1.b bVar = e1.b.RAW;
            e1.a aVar = e1.a.ANALYSIS;
            e1.b bVar2 = e1.b.JPEG;
            e1.a aVar2 = e1.a.PREVIEW;
            e1.b bVar3 = e1.b.YUV;
            e1.a aVar3 = e1.a.MAXIMUM;
            e1.b bVar4 = e1.b.PRIV;
            ArrayList arrayList2 = new ArrayList();
            b.d.b.d1.d1 d1Var = new b.d.b.d1.d1();
            b.d.b.d1.d1 O = c.b.a.a.a.O(d1Var.f1443a, new b.d.b.d1.o(bVar4, aVar3), arrayList2, d1Var);
            b.d.b.d1.d1 O2 = c.b.a.a.a.O(O.f1443a, new b.d.b.d1.o(bVar2, aVar3), arrayList2, O);
            b.d.b.d1.d1 O3 = c.b.a.a.a.O(O2.f1443a, new b.d.b.d1.o(bVar3, aVar3), arrayList2, O2);
            b.d.b.d1.d1 O4 = c.b.a.a.a.O(O3.f1443a, c.b.a.a.a.H(O3.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar2, aVar3), arrayList2, O3);
            b.d.b.d1.d1 O5 = c.b.a.a.a.O(O4.f1443a, c.b.a.a.a.H(O4.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar2, aVar3), arrayList2, O4);
            b.d.b.d1.d1 O6 = c.b.a.a.a.O(O5.f1443a, c.b.a.a.a.H(O5.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar2), arrayList2, O5);
            b.d.b.d1.d1 O7 = c.b.a.a.a.O(O6.f1443a, c.b.a.a.a.H(O6.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar2), arrayList2, O6);
            O7.f1443a.add(c.b.a.a.a.H(O7.f1443a, c.b.a.a.a.H(O7.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar2), bVar2, aVar3));
            arrayList2.add(O7);
            arrayList.addAll(arrayList2);
            if (intValue == 0 || intValue == 1 || intValue == 3) {
                ArrayList arrayList3 = new ArrayList();
                b.d.b.d1.d1 d1Var2 = new b.d.b.d1.d1();
                d1Var2.f1443a.add(new b.d.b.d1.o(bVar4, aVar2));
                e1.a aVar4 = e1.a.RECORD;
                windowManager = windowManager2;
                b.d.b.d1.d1 O8 = c.b.a.a.a.O(d1Var2.f1443a, new b.d.b.d1.o(bVar4, aVar4), arrayList3, d1Var2);
                b.d.b.d1.d1 O9 = c.b.a.a.a.O(O8.f1443a, c.b.a.a.a.H(O8.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar4), arrayList3, O8);
                b.d.b.d1.d1 O10 = c.b.a.a.a.O(O9.f1443a, c.b.a.a.a.H(O9.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar3, aVar4), arrayList3, O9);
                b.d.b.d1.d1 O11 = c.b.a.a.a.O(O10.f1443a, c.b.a.a.a.H(O10.f1443a, c.b.a.a.a.H(O10.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar4), bVar2, aVar4), arrayList3, O10);
                b.d.b.d1.d1 O12 = c.b.a.a.a.O(O11.f1443a, c.b.a.a.a.H(O11.f1443a, c.b.a.a.a.H(O11.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar4), bVar2, aVar4), arrayList3, O11);
                O12.f1443a.add(c.b.a.a.a.H(O12.f1443a, c.b.a.a.a.H(O12.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar3, aVar2), bVar2, aVar3));
                arrayList3.add(O12);
                arrayList.addAll(arrayList3);
            } else {
                windowManager = windowManager2;
            }
            if (intValue == 1 || intValue == 3) {
                ArrayList arrayList4 = new ArrayList();
                b.d.b.d1.d1 d1Var3 = new b.d.b.d1.d1();
                b.d.b.d1.d1 O13 = c.b.a.a.a.O(d1Var3.f1443a, c.b.a.a.a.H(d1Var3.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar3), arrayList4, d1Var3);
                b.d.b.d1.d1 O14 = c.b.a.a.a.O(O13.f1443a, c.b.a.a.a.H(O13.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar3), arrayList4, O13);
                b.d.b.d1.d1 O15 = c.b.a.a.a.O(O14.f1443a, c.b.a.a.a.H(O14.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar3, aVar3), arrayList4, O14);
                b.d.b.d1.d1 O16 = c.b.a.a.a.O(O15.f1443a, c.b.a.a.a.H(O15.f1443a, c.b.a.a.a.H(O15.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar2), bVar2, aVar3), arrayList4, O15);
                b.d.b.d1.d1 O17 = c.b.a.a.a.O(O16.f1443a, c.b.a.a.a.H(O16.f1443a, c.b.a.a.a.H(O16.f1443a, new b.d.b.d1.o(bVar3, aVar), bVar4, aVar2), bVar3, aVar3), arrayList4, O16);
                O17.f1443a.add(c.b.a.a.a.H(O17.f1443a, c.b.a.a.a.H(O17.f1443a, new b.d.b.d1.o(bVar3, aVar), bVar3, aVar2), bVar3, aVar3));
                arrayList4.add(O17);
                arrayList.addAll(arrayList4);
            }
            int[] iArr = (int[]) b2.a(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
            if (iArr != null) {
                for (int i2 : iArr) {
                    if (i2 == 3) {
                        this.u = true;
                    } else if (i2 == 6) {
                        this.v = true;
                    }
                }
            }
            if (this.u) {
                List<b.d.b.d1.d1> list = this.l;
                ArrayList arrayList5 = new ArrayList();
                b.d.b.d1.d1 d1Var4 = new b.d.b.d1.d1();
                b.d.b.d1.d1 O18 = c.b.a.a.a.O(d1Var4.f1443a, new b.d.b.d1.o(bVar, aVar3), arrayList5, d1Var4);
                b.d.b.d1.d1 O19 = c.b.a.a.a.O(O18.f1443a, c.b.a.a.a.H(O18.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar, aVar3), arrayList5, O18);
                b.d.b.d1.d1 O20 = c.b.a.a.a.O(O19.f1443a, c.b.a.a.a.H(O19.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar, aVar3), arrayList5, O19);
                b.d.b.d1.d1 O21 = c.b.a.a.a.O(O20.f1443a, c.b.a.a.a.H(O20.f1443a, c.b.a.a.a.H(O20.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar2), bVar, aVar3), arrayList5, O20);
                b.d.b.d1.d1 O22 = c.b.a.a.a.O(O21.f1443a, c.b.a.a.a.H(O21.f1443a, c.b.a.a.a.H(O21.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar2), bVar, aVar3), arrayList5, O21);
                b.d.b.d1.d1 O23 = c.b.a.a.a.O(O22.f1443a, c.b.a.a.a.H(O22.f1443a, c.b.a.a.a.H(O22.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar3, aVar2), bVar, aVar3), arrayList5, O22);
                b.d.b.d1.d1 O24 = c.b.a.a.a.O(O23.f1443a, c.b.a.a.a.H(O23.f1443a, c.b.a.a.a.H(O23.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar2, aVar3), bVar, aVar3), arrayList5, O23);
                O24.f1443a.add(c.b.a.a.a.H(O24.f1443a, c.b.a.a.a.H(O24.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar2, aVar3), bVar, aVar3));
                arrayList5.add(O24);
                list.addAll(arrayList5);
            }
            if (this.v && this.r == 0) {
                List<b.d.b.d1.d1> list2 = this.l;
                ArrayList arrayList6 = new ArrayList();
                b.d.b.d1.d1 d1Var5 = new b.d.b.d1.d1();
                b.d.b.d1.d1 O25 = c.b.a.a.a.O(d1Var5.f1443a, c.b.a.a.a.H(d1Var5.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar3), arrayList6, d1Var5);
                b.d.b.d1.d1 O26 = c.b.a.a.a.O(O25.f1443a, c.b.a.a.a.H(O25.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar3, aVar3), arrayList6, O25);
                O26.f1443a.add(c.b.a.a.a.H(O26.f1443a, new b.d.b.d1.o(bVar3, aVar2), bVar3, aVar3));
                arrayList6.add(O26);
                list2.addAll(arrayList6);
            }
            if (this.r == 3) {
                List<b.d.b.d1.d1> list3 = this.l;
                ArrayList arrayList7 = new ArrayList();
                b.d.b.d1.d1 d1Var6 = new b.d.b.d1.d1();
                b.d.b.d1.d1 O27 = c.b.a.a.a.O(d1Var6.f1443a, c.b.a.a.a.H(d1Var6.f1443a, c.b.a.a.a.H(d1Var6.f1443a, c.b.a.a.a.H(d1Var6.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar), bVar3, aVar3), bVar, aVar3), arrayList7, d1Var6);
                O27.f1443a.add(c.b.a.a.a.H(O27.f1443a, c.b.a.a.a.H(O27.f1443a, c.b.a.a.a.H(O27.f1443a, new b.d.b.d1.o(bVar4, aVar2), bVar4, aVar), bVar2, aVar3), bVar, aVar3));
                arrayList7.add(O27);
                list3.addAll(arrayList7);
            }
            Size size3 = new Size(640, 480);
            Point point = new Point();
            windowManager.getDefaultDisplay().getRealSize(point);
            if (point.x > point.y) {
                size = new Size(point.x, point.y);
            } else {
                size = new Size(point.y, point.x);
            }
            Size size4 = new Size(size.getWidth(), size.getHeight());
            int i3 = 0;
            Size size5 = (Size) Collections.min(Arrays.asList(size4, f1122a), new b());
            Size size6 = f1128g;
            try {
                int parseInt = Integer.parseInt(this.n);
                if (this.o.a(parseInt, 8)) {
                    size6 = f1125d;
                } else if (this.o.a(parseInt, 6)) {
                    size6 = f1126e;
                } else if (this.o.a(parseInt, 5)) {
                    size6 = f1127f;
                } else {
                    this.o.a(parseInt, 4);
                }
            } catch (NumberFormatException unused) {
                StreamConfigurationMap streamConfigurationMap = (StreamConfigurationMap) this.p.a(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (streamConfigurationMap != null) {
                    Size[] outputSizes = streamConfigurationMap.getOutputSizes(MediaRecorder.class);
                    if (outputSizes == null) {
                        size6 = f1128g;
                    } else {
                        Arrays.sort(outputSizes, new b(true));
                        int length = outputSizes.length;
                        while (true) {
                            if (i3 < length) {
                                Size size7 = outputSizes[i3];
                                int width = size7.getWidth();
                                Size size8 = f1126e;
                                if (width <= size8.getWidth() && size7.getHeight() <= size8.getHeight()) {
                                    size6 = size7;
                                    break;
                                }
                                i3++;
                            } else {
                                size6 = f1128g;
                                break;
                            }
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Can not retrieve SCALER_STREAM_CONFIGURATION_MAP");
                }
            }
            this.w = new b.d.b.d1.p(size3, size5, size6);
        } catch (b.d.a.e.y1.a e2) {
            throw b.b.a.d(e2);
        }
    }

    public static int e(Size size) {
        return size.getHeight() * size.getWidth();
    }

    public static boolean g(int i2, int i3, Rational rational) {
        b.j.b.d.d(i3 % 16 == 0);
        double numerator = (rational.getNumerator() * i2) / rational.getDenominator();
        return numerator > ((double) Math.max(0, i3 + (-16))) && numerator < ((double) (i3 + 16));
    }

    /* JADX WARN: Code restructure failed: missing block: B:42:0x0096, code lost:
        continue;
     */
    /* JADX WARN: Removed duplicated region for block: B:36:0x009d A[EDGE_INSN: B:36:0x009d->B:34:0x009d ?: BREAK  , SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean a(List<b.d.b.d1.e1> list) {
        boolean z = false;
        for (b.d.b.d1.d1 d1Var : this.l) {
            Objects.requireNonNull(d1Var);
            boolean z2 = true;
            if (!list.isEmpty()) {
                if (list.size() > d1Var.f1443a.size()) {
                    z = false;
                    continue;
                    if (z) {
                        break;
                    }
                } else {
                    int size = d1Var.f1443a.size();
                    ArrayList arrayList = new ArrayList();
                    b.d.b.d1.d1.a(arrayList, size, new int[size], 0);
                    Iterator it = arrayList.iterator();
                    while (true) {
                        if (!it.hasNext()) {
                            z2 = false;
                            break;
                        }
                        int[] iArr = (int[]) it.next();
                        boolean z3 = true;
                        for (int i2 = 0; i2 < d1Var.f1443a.size(); i2++) {
                            if (iArr[i2] < list.size()) {
                                b.d.b.d1.e1 e1Var = d1Var.f1443a.get(i2);
                                b.d.b.d1.e1 e1Var2 = list.get(iArr[i2]);
                                Objects.requireNonNull(e1Var);
                                z3 &= e1Var2.a().f1451h <= e1Var.a().f1451h && e1Var2.b() == e1Var.b();
                                if (!z3) {
                                    break;
                                }
                            }
                        }
                        if (z3) {
                            break;
                        }
                    }
                }
            }
            z = z2;
            continue;
            if (z) {
            }
        }
        return z;
    }

    public final Size[] b(Size[] sizeArr, int i2) {
        ArrayList arrayList;
        List<Size> list = this.t.get(Integer.valueOf(i2));
        if (list == null) {
            b.d.a.e.y1.q.c cVar = this.q;
            Objects.requireNonNull(cVar);
            if (((b.d.a.e.y1.p.e) b.d.a.e.y1.p.d.a(b.d.a.e.y1.p.e.class)) == null) {
                list = new ArrayList<>();
            } else {
                String str = cVar.f1353a;
                if (b.d.a.e.y1.p.e.a()) {
                    ArrayList arrayList2 = new ArrayList();
                    arrayList = arrayList2;
                    arrayList = arrayList2;
                    if (str.equals(CrashlyticsReportDataCapture.SIGNAL_DEFAULT) && i2 == 256) {
                        arrayList2.add(new Size(4160, 3120));
                        arrayList2.add(new Size(4000, 3000));
                        arrayList = arrayList2;
                    }
                } else if (b.d.a.e.y1.p.e.b()) {
                    ArrayList arrayList3 = new ArrayList();
                    arrayList = arrayList3;
                    arrayList = arrayList3;
                    if (str.equals(CrashlyticsReportDataCapture.SIGNAL_DEFAULT) && i2 == 256) {
                        arrayList3.add(new Size(4160, 3120));
                        arrayList3.add(new Size(4000, 3000));
                        arrayList = arrayList3;
                    }
                } else {
                    b.d.b.u0.d("ExcludedSupportedSizesQuirk", "Cannot retrieve list of supported sizes to exclude on this device.", null);
                    arrayList = Collections.emptyList();
                }
                list = arrayList;
            }
            this.t.put(Integer.valueOf(i2), list);
        }
        ArrayList arrayList4 = new ArrayList(Arrays.asList(sizeArr));
        arrayList4.removeAll(list);
        return (Size[]) arrayList4.toArray(new Size[0]);
    }

    public final Size c(int i2) {
        Size size = this.m.get(Integer.valueOf(i2));
        if (size != null) {
            return size;
        }
        Size size2 = (Size) Collections.max(Arrays.asList(d(i2)), new b());
        this.m.put(Integer.valueOf(i2), size2);
        return size2;
    }

    public final Size[] d(int i2) {
        Size[] sizeArr = this.x.get(Integer.valueOf(i2));
        if (sizeArr == null) {
            StreamConfigurationMap streamConfigurationMap = (StreamConfigurationMap) this.p.a(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (streamConfigurationMap != null) {
                Size[] outputSizes = streamConfigurationMap.getOutputSizes(i2);
                if (outputSizes != null) {
                    Size[] b2 = b(outputSizes, i2);
                    Arrays.sort(b2, new b(true));
                    this.x.put(Integer.valueOf(i2), b2);
                    return b2;
                }
                throw new IllegalArgumentException(c.b.a.a.a.j("Can not get supported output size for the format: ", i2));
            }
            throw new IllegalArgumentException("Can not retrieve SCALER_STREAM_CONFIGURATION_MAP");
        }
        return sizeArr;
    }

    public final Size f(b.d.b.d1.n0 n0Var) {
        boolean z = false;
        int w = n0Var.w(0);
        Size o = n0Var.o(null);
        if (o != null) {
            Integer num = (Integer) this.p.a(CameraCharacteristics.SENSOR_ORIENTATION);
            b.j.b.d.h(num, "Camera HAL in bad state, unable to retrieve the SENSOR_ORIENTATION");
            int o2 = b.b.a.o(w);
            Integer num2 = (Integer) this.p.a(CameraCharacteristics.LENS_FACING);
            b.j.b.d.h(num2, "Camera HAL in bad state, unable to retrieve the LENS_FACING");
            int i2 = b.b.a.i(o2, num.intValue(), 1 == num2.intValue());
            if (i2 == 90 || i2 == 270) {
                z = true;
            }
            return z ? new Size(o.getHeight(), o.getWidth()) : o;
        }
        return o;
    }

    public final void h(List<Size> list, Size size) {
        if (list == null || list.isEmpty()) {
            return;
        }
        int i2 = -1;
        ArrayList arrayList = new ArrayList();
        int i3 = 0;
        while (true) {
            int i4 = i3;
            int i5 = i2;
            i2 = i4;
            if (i2 >= list.size()) {
                break;
            }
            Size size2 = list.get(i2);
            if (size2.getWidth() < size.getWidth() || size2.getHeight() < size.getHeight()) {
                break;
            }
            if (i5 >= 0) {
                arrayList.add(list.get(i5));
            }
            i3 = i2 + 1;
        }
        list.removeAll(arrayList);
    }

    public b.d.b.d1.e1 i(int i2, Size size) {
        e1.b bVar;
        e1.a aVar = e1.a.NOT_SUPPORT;
        if (i2 == 35) {
            bVar = e1.b.YUV;
        } else if (i2 == 256) {
            bVar = e1.b.JPEG;
        } else if (i2 == 32) {
            bVar = e1.b.RAW;
        } else {
            bVar = e1.b.PRIV;
        }
        Size c2 = c(i2);
        if (size.getHeight() * size.getWidth() <= this.w.a().getHeight() * this.w.a().getWidth()) {
            aVar = e1.a.ANALYSIS;
        } else {
            if (size.getHeight() * size.getWidth() <= this.w.b().getHeight() * this.w.b().getWidth()) {
                aVar = e1.a.PREVIEW;
            } else {
                if (size.getHeight() * size.getWidth() <= this.w.c().getHeight() * this.w.c().getWidth()) {
                    aVar = e1.a.RECORD;
                } else {
                    if (size.getHeight() * size.getWidth() <= c2.getHeight() * c2.getWidth()) {
                        aVar = e1.a.MAXIMUM;
                    }
                }
            }
        }
        return new b.d.b.d1.o(bVar, aVar);
    }

    /* compiled from: SupportedSurfaceCombination.java */
    /* loaded from: classes.dex */
    public static final class b implements Comparator<Size> {

        /* renamed from: b  reason: collision with root package name */
        public boolean f1131b;

        public b() {
            this.f1131b = false;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(Size size, Size size2) {
            Size size3 = size;
            Size size4 = size2;
            int signum = Long.signum((size3.getWidth() * size3.getHeight()) - (size4.getWidth() * size4.getHeight()));
            return this.f1131b ? signum * (-1) : signum;
        }

        public b(boolean z) {
            this.f1131b = false;
            this.f1131b = z;
        }
    }
}