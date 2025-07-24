package b.h.c;

import android.content.Context;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseIntArray;
import android.util.Xml;
import android.view.View;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.Barrier;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import b.d.b.m0;
import b.h.c.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Objects;
import org.xmlpull.v1.XmlPullParserException;

/* compiled from: ConstraintSet.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public static final int[] f1965a = {0, 4, 8};

    /* renamed from: b  reason: collision with root package name */
    public static SparseIntArray f1966b;

    /* renamed from: c  reason: collision with root package name */
    public HashMap<String, b.h.c.a> f1967c = new HashMap<>();

    /* renamed from: d  reason: collision with root package name */
    public boolean f1968d = true;

    /* renamed from: e  reason: collision with root package name */
    public HashMap<Integer, a> f1969e = new HashMap<>();

    /* compiled from: ConstraintSet.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public int f1970a;

        /* renamed from: b  reason: collision with root package name */
        public final C0030d f1971b = new C0030d();

        /* renamed from: c  reason: collision with root package name */
        public final c f1972c = new c();

        /* renamed from: d  reason: collision with root package name */
        public final b f1973d = new b();

        /* renamed from: e  reason: collision with root package name */
        public final e f1974e = new e();

        /* renamed from: f  reason: collision with root package name */
        public HashMap<String, b.h.c.a> f1975f = new HashMap<>();

        public void a(ConstraintLayout.a aVar) {
            b bVar = this.f1973d;
            aVar.f200d = bVar.i;
            aVar.f201e = bVar.j;
            aVar.f202f = bVar.k;
            aVar.f203g = bVar.l;
            aVar.f204h = bVar.m;
            aVar.i = bVar.n;
            aVar.j = bVar.o;
            aVar.k = bVar.p;
            aVar.l = bVar.q;
            aVar.p = bVar.r;
            aVar.q = bVar.s;
            aVar.r = bVar.t;
            aVar.s = bVar.u;
            ((ViewGroup.MarginLayoutParams) aVar).leftMargin = bVar.E;
            ((ViewGroup.MarginLayoutParams) aVar).rightMargin = bVar.F;
            ((ViewGroup.MarginLayoutParams) aVar).topMargin = bVar.G;
            ((ViewGroup.MarginLayoutParams) aVar).bottomMargin = bVar.H;
            aVar.x = bVar.P;
            aVar.y = bVar.O;
            aVar.u = bVar.L;
            aVar.w = bVar.N;
            aVar.z = bVar.v;
            aVar.A = bVar.w;
            aVar.m = bVar.y;
            aVar.n = bVar.z;
            b bVar2 = this.f1973d;
            aVar.o = bVar2.A;
            aVar.B = bVar2.x;
            aVar.P = bVar2.B;
            aVar.Q = bVar2.C;
            aVar.E = bVar2.Q;
            aVar.D = bVar2.R;
            aVar.G = bVar2.T;
            aVar.F = bVar2.S;
            aVar.S = bVar2.i0;
            aVar.T = bVar2.j0;
            aVar.H = bVar2.U;
            aVar.I = bVar2.V;
            aVar.L = bVar2.W;
            aVar.M = bVar2.X;
            aVar.J = bVar2.Y;
            aVar.K = bVar2.Z;
            aVar.N = bVar2.a0;
            aVar.O = bVar2.b0;
            aVar.R = bVar2.D;
            aVar.f199c = bVar2.f1983h;
            aVar.f197a = bVar2.f1981f;
            aVar.f198b = bVar2.f1982g;
            ((ViewGroup.MarginLayoutParams) aVar).width = bVar2.f1979d;
            ((ViewGroup.MarginLayoutParams) aVar).height = bVar2.f1980e;
            String str = bVar2.h0;
            if (str != null) {
                aVar.U = str;
            }
            aVar.setMarginStart(this.f1973d.J);
            aVar.setMarginEnd(this.f1973d.I);
            aVar.a();
        }

        public final void b(int i, ConstraintLayout.a aVar) {
            this.f1970a = i;
            b bVar = this.f1973d;
            bVar.i = aVar.f200d;
            bVar.j = aVar.f201e;
            bVar.k = aVar.f202f;
            bVar.l = aVar.f203g;
            bVar.m = aVar.f204h;
            bVar.n = aVar.i;
            bVar.o = aVar.j;
            bVar.p = aVar.k;
            bVar.q = aVar.l;
            bVar.r = aVar.p;
            bVar.s = aVar.q;
            bVar.t = aVar.r;
            bVar.u = aVar.s;
            bVar.v = aVar.z;
            bVar.w = aVar.A;
            bVar.x = aVar.B;
            bVar.y = aVar.m;
            bVar.z = aVar.n;
            bVar.A = aVar.o;
            bVar.B = aVar.P;
            bVar.C = aVar.Q;
            bVar.D = aVar.R;
            bVar.f1983h = aVar.f199c;
            bVar.f1981f = aVar.f197a;
            bVar.f1982g = aVar.f198b;
            b bVar2 = this.f1973d;
            bVar2.f1979d = ((ViewGroup.MarginLayoutParams) aVar).width;
            bVar2.f1980e = ((ViewGroup.MarginLayoutParams) aVar).height;
            bVar2.E = ((ViewGroup.MarginLayoutParams) aVar).leftMargin;
            bVar2.F = ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
            bVar2.G = ((ViewGroup.MarginLayoutParams) aVar).topMargin;
            bVar2.H = ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
            bVar2.Q = aVar.E;
            bVar2.R = aVar.D;
            bVar2.T = aVar.G;
            bVar2.S = aVar.F;
            bVar2.i0 = aVar.S;
            bVar2.j0 = aVar.T;
            bVar2.U = aVar.H;
            bVar2.V = aVar.I;
            bVar2.W = aVar.L;
            bVar2.X = aVar.M;
            bVar2.Y = aVar.J;
            bVar2.Z = aVar.K;
            bVar2.a0 = aVar.N;
            bVar2.b0 = aVar.O;
            bVar2.h0 = aVar.U;
            bVar2.L = aVar.u;
            bVar2.N = aVar.w;
            bVar2.K = aVar.t;
            bVar2.M = aVar.v;
            b bVar3 = this.f1973d;
            bVar3.P = aVar.x;
            bVar3.O = aVar.y;
            bVar3.I = aVar.getMarginEnd();
            this.f1973d.J = aVar.getMarginStart();
        }

        public final void c(int i, e.a aVar) {
            b(i, aVar);
            this.f1971b.f1995d = aVar.m0;
            e eVar = this.f1974e;
            eVar.f1999c = aVar.p0;
            eVar.f2000d = aVar.q0;
            eVar.f2001e = aVar.r0;
            eVar.f2002f = aVar.s0;
            eVar.f2003g = aVar.t0;
            eVar.f2004h = aVar.u0;
            eVar.i = aVar.v0;
            eVar.j = aVar.w0;
            eVar.k = aVar.x0;
            eVar.l = aVar.y0;
            eVar.n = aVar.o0;
            eVar.m = aVar.n0;
        }

        public Object clone() {
            a aVar = new a();
            b bVar = aVar.f1973d;
            b bVar2 = this.f1973d;
            Objects.requireNonNull(bVar);
            bVar.f1977b = bVar2.f1977b;
            bVar.f1979d = bVar2.f1979d;
            bVar.f1978c = bVar2.f1978c;
            bVar.f1980e = bVar2.f1980e;
            bVar.f1981f = bVar2.f1981f;
            bVar.f1982g = bVar2.f1982g;
            bVar.f1983h = bVar2.f1983h;
            bVar.i = bVar2.i;
            bVar.j = bVar2.j;
            bVar.k = bVar2.k;
            bVar.l = bVar2.l;
            bVar.m = bVar2.m;
            bVar.n = bVar2.n;
            bVar.o = bVar2.o;
            bVar.p = bVar2.p;
            bVar.q = bVar2.q;
            bVar.r = bVar2.r;
            bVar.s = bVar2.s;
            bVar.t = bVar2.t;
            bVar.u = bVar2.u;
            bVar.v = bVar2.v;
            bVar.w = bVar2.w;
            bVar.x = bVar2.x;
            bVar.y = bVar2.y;
            bVar.z = bVar2.z;
            bVar.A = bVar2.A;
            bVar.B = bVar2.B;
            bVar.C = bVar2.C;
            bVar.D = bVar2.D;
            bVar.E = bVar2.E;
            bVar.F = bVar2.F;
            bVar.G = bVar2.G;
            bVar.H = bVar2.H;
            bVar.I = bVar2.I;
            bVar.J = bVar2.J;
            bVar.K = bVar2.K;
            bVar.L = bVar2.L;
            bVar.M = bVar2.M;
            bVar.N = bVar2.N;
            bVar.O = bVar2.O;
            bVar.P = bVar2.P;
            bVar.Q = bVar2.Q;
            bVar.R = bVar2.R;
            bVar.S = bVar2.S;
            bVar.T = bVar2.T;
            bVar.U = bVar2.U;
            bVar.V = bVar2.V;
            bVar.W = bVar2.W;
            bVar.X = bVar2.X;
            bVar.Y = bVar2.Y;
            bVar.Z = bVar2.Z;
            bVar.a0 = bVar2.a0;
            bVar.b0 = bVar2.b0;
            bVar.c0 = bVar2.c0;
            bVar.d0 = bVar2.d0;
            bVar.e0 = bVar2.e0;
            bVar.h0 = bVar2.h0;
            int[] iArr = bVar2.f0;
            if (iArr != null) {
                bVar.f0 = Arrays.copyOf(iArr, iArr.length);
            } else {
                bVar.f0 = null;
            }
            bVar.g0 = bVar2.g0;
            bVar.i0 = bVar2.i0;
            bVar.j0 = bVar2.j0;
            bVar.k0 = bVar2.k0;
            c cVar = aVar.f1972c;
            c cVar2 = this.f1972c;
            Objects.requireNonNull(cVar);
            cVar.f1985b = cVar2.f1985b;
            cVar.f1986c = cVar2.f1986c;
            cVar.f1987d = cVar2.f1987d;
            cVar.f1988e = cVar2.f1988e;
            cVar.f1989f = cVar2.f1989f;
            cVar.f1991h = cVar2.f1991h;
            cVar.f1990g = cVar2.f1990g;
            C0030d c0030d = aVar.f1971b;
            C0030d c0030d2 = this.f1971b;
            Objects.requireNonNull(c0030d);
            c0030d.f1992a = c0030d2.f1992a;
            c0030d.f1993b = c0030d2.f1993b;
            c0030d.f1995d = c0030d2.f1995d;
            c0030d.f1996e = c0030d2.f1996e;
            c0030d.f1994c = c0030d2.f1994c;
            e eVar = aVar.f1974e;
            e eVar2 = this.f1974e;
            Objects.requireNonNull(eVar);
            eVar.f1998b = eVar2.f1998b;
            eVar.f1999c = eVar2.f1999c;
            eVar.f2000d = eVar2.f2000d;
            eVar.f2001e = eVar2.f2001e;
            eVar.f2002f = eVar2.f2002f;
            eVar.f2003g = eVar2.f2003g;
            eVar.f2004h = eVar2.f2004h;
            eVar.i = eVar2.i;
            eVar.j = eVar2.j;
            eVar.k = eVar2.k;
            eVar.l = eVar2.l;
            eVar.m = eVar2.m;
            eVar.n = eVar2.n;
            aVar.f1970a = this.f1970a;
            return aVar;
        }
    }

    /* compiled from: ConstraintSet.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public static SparseIntArray f1976a;

        /* renamed from: d  reason: collision with root package name */
        public int f1979d;

        /* renamed from: e  reason: collision with root package name */
        public int f1980e;
        public int[] f0;
        public String g0;
        public String h0;

        /* renamed from: b  reason: collision with root package name */
        public boolean f1977b = false;

        /* renamed from: c  reason: collision with root package name */
        public boolean f1978c = false;

        /* renamed from: f  reason: collision with root package name */
        public int f1981f = -1;

        /* renamed from: g  reason: collision with root package name */
        public int f1982g = -1;

        /* renamed from: h  reason: collision with root package name */
        public float f1983h = -1.0f;
        public int i = -1;
        public int j = -1;
        public int k = -1;
        public int l = -1;
        public int m = -1;
        public int n = -1;
        public int o = -1;
        public int p = -1;
        public int q = -1;
        public int r = -1;
        public int s = -1;
        public int t = -1;
        public int u = -1;
        public float v = 0.5f;
        public float w = 0.5f;
        public String x = null;
        public int y = -1;
        public int z = 0;
        public float A = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public int B = -1;
        public int C = -1;
        public int D = -1;
        public int E = -1;
        public int F = -1;
        public int G = -1;
        public int H = -1;
        public int I = -1;
        public int J = -1;
        public int K = -1;
        public int L = -1;
        public int M = -1;
        public int N = -1;
        public int O = -1;
        public int P = -1;
        public float Q = -1.0f;
        public float R = -1.0f;
        public int S = 0;
        public int T = 0;
        public int U = 0;
        public int V = 0;
        public int W = -1;
        public int X = -1;
        public int Y = -1;
        public int Z = -1;
        public float a0 = 1.0f;
        public float b0 = 1.0f;
        public int c0 = -1;
        public int d0 = 0;
        public int e0 = -1;
        public boolean i0 = false;
        public boolean j0 = false;
        public boolean k0 = true;

        static {
            SparseIntArray sparseIntArray = new SparseIntArray();
            f1976a = sparseIntArray;
            sparseIntArray.append(39, 24);
            f1976a.append(40, 25);
            f1976a.append(42, 28);
            f1976a.append(43, 29);
            f1976a.append(48, 35);
            f1976a.append(47, 34);
            f1976a.append(21, 4);
            f1976a.append(20, 3);
            f1976a.append(18, 1);
            f1976a.append(56, 6);
            f1976a.append(57, 7);
            f1976a.append(28, 17);
            f1976a.append(29, 18);
            f1976a.append(30, 19);
            f1976a.append(0, 26);
            f1976a.append(44, 31);
            f1976a.append(45, 32);
            f1976a.append(27, 10);
            f1976a.append(26, 9);
            f1976a.append(60, 13);
            f1976a.append(63, 16);
            f1976a.append(61, 14);
            f1976a.append(58, 11);
            f1976a.append(62, 15);
            f1976a.append(59, 12);
            f1976a.append(51, 38);
            f1976a.append(37, 37);
            f1976a.append(36, 39);
            f1976a.append(50, 40);
            f1976a.append(35, 20);
            f1976a.append(49, 36);
            f1976a.append(25, 5);
            f1976a.append(38, 76);
            f1976a.append(46, 76);
            f1976a.append(41, 76);
            f1976a.append(19, 76);
            f1976a.append(17, 76);
            f1976a.append(3, 23);
            f1976a.append(5, 27);
            f1976a.append(7, 30);
            f1976a.append(8, 8);
            f1976a.append(4, 33);
            f1976a.append(6, 2);
            f1976a.append(1, 22);
            f1976a.append(2, 21);
            f1976a.append(22, 61);
            f1976a.append(24, 62);
            f1976a.append(23, 63);
            f1976a.append(55, 69);
            f1976a.append(34, 70);
            f1976a.append(12, 71);
            f1976a.append(10, 72);
            f1976a.append(11, 73);
            f1976a.append(13, 74);
            f1976a.append(9, 75);
        }

        public void a(Context context, AttributeSet attributeSet) {
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2013e);
            this.f1978c = true;
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                int i2 = f1976a.get(index);
                if (i2 == 80) {
                    this.i0 = obtainStyledAttributes.getBoolean(index, this.i0);
                } else if (i2 != 81) {
                    switch (i2) {
                        case 1:
                            int i3 = this.q;
                            int[] iArr = d.f1965a;
                            int resourceId = obtainStyledAttributes.getResourceId(index, i3);
                            if (resourceId == -1) {
                                resourceId = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.q = resourceId;
                            continue;
                        case 2:
                            this.H = obtainStyledAttributes.getDimensionPixelSize(index, this.H);
                            continue;
                        case 3:
                            int i4 = this.p;
                            int[] iArr2 = d.f1965a;
                            int resourceId2 = obtainStyledAttributes.getResourceId(index, i4);
                            if (resourceId2 == -1) {
                                resourceId2 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.p = resourceId2;
                            continue;
                        case 4:
                            int i5 = this.o;
                            int[] iArr3 = d.f1965a;
                            int resourceId3 = obtainStyledAttributes.getResourceId(index, i5);
                            if (resourceId3 == -1) {
                                resourceId3 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.o = resourceId3;
                            continue;
                        case 5:
                            this.x = obtainStyledAttributes.getString(index);
                            continue;
                        case 6:
                            this.B = obtainStyledAttributes.getDimensionPixelOffset(index, this.B);
                            continue;
                        case 7:
                            this.C = obtainStyledAttributes.getDimensionPixelOffset(index, this.C);
                            continue;
                        case 8:
                            this.I = obtainStyledAttributes.getDimensionPixelSize(index, this.I);
                            continue;
                        case 9:
                            int i6 = this.u;
                            int[] iArr4 = d.f1965a;
                            int resourceId4 = obtainStyledAttributes.getResourceId(index, i6);
                            if (resourceId4 == -1) {
                                resourceId4 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.u = resourceId4;
                            continue;
                        case 10:
                            int i7 = this.t;
                            int[] iArr5 = d.f1965a;
                            int resourceId5 = obtainStyledAttributes.getResourceId(index, i7);
                            if (resourceId5 == -1) {
                                resourceId5 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.t = resourceId5;
                            continue;
                        case 11:
                            this.N = obtainStyledAttributes.getDimensionPixelSize(index, this.N);
                            continue;
                        case 12:
                            this.O = obtainStyledAttributes.getDimensionPixelSize(index, this.O);
                            continue;
                        case 13:
                            this.K = obtainStyledAttributes.getDimensionPixelSize(index, this.K);
                            continue;
                        case 14:
                            this.M = obtainStyledAttributes.getDimensionPixelSize(index, this.M);
                            continue;
                        case 15:
                            this.P = obtainStyledAttributes.getDimensionPixelSize(index, this.P);
                            continue;
                        case 16:
                            this.L = obtainStyledAttributes.getDimensionPixelSize(index, this.L);
                            continue;
                        case 17:
                            this.f1981f = obtainStyledAttributes.getDimensionPixelOffset(index, this.f1981f);
                            continue;
                        case 18:
                            this.f1982g = obtainStyledAttributes.getDimensionPixelOffset(index, this.f1982g);
                            continue;
                        case 19:
                            this.f1983h = obtainStyledAttributes.getFloat(index, this.f1983h);
                            continue;
                        case 20:
                            this.v = obtainStyledAttributes.getFloat(index, this.v);
                            continue;
                        case 21:
                            this.f1980e = obtainStyledAttributes.getLayoutDimension(index, this.f1980e);
                            continue;
                        case 22:
                            this.f1979d = obtainStyledAttributes.getLayoutDimension(index, this.f1979d);
                            continue;
                        case 23:
                            this.E = obtainStyledAttributes.getDimensionPixelSize(index, this.E);
                            continue;
                        case 24:
                            int i8 = this.i;
                            int[] iArr6 = d.f1965a;
                            int resourceId6 = obtainStyledAttributes.getResourceId(index, i8);
                            if (resourceId6 == -1) {
                                resourceId6 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.i = resourceId6;
                            continue;
                        case 25:
                            int i9 = this.j;
                            int[] iArr7 = d.f1965a;
                            int resourceId7 = obtainStyledAttributes.getResourceId(index, i9);
                            if (resourceId7 == -1) {
                                resourceId7 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.j = resourceId7;
                            continue;
                        case 26:
                            this.D = obtainStyledAttributes.getInt(index, this.D);
                            continue;
                        case 27:
                            this.F = obtainStyledAttributes.getDimensionPixelSize(index, this.F);
                            continue;
                        case 28:
                            int i10 = this.k;
                            int[] iArr8 = d.f1965a;
                            int resourceId8 = obtainStyledAttributes.getResourceId(index, i10);
                            if (resourceId8 == -1) {
                                resourceId8 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.k = resourceId8;
                            continue;
                        case 29:
                            int i11 = this.l;
                            int[] iArr9 = d.f1965a;
                            int resourceId9 = obtainStyledAttributes.getResourceId(index, i11);
                            if (resourceId9 == -1) {
                                resourceId9 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.l = resourceId9;
                            continue;
                        case 30:
                            this.J = obtainStyledAttributes.getDimensionPixelSize(index, this.J);
                            continue;
                        case 31:
                            int i12 = this.r;
                            int[] iArr10 = d.f1965a;
                            int resourceId10 = obtainStyledAttributes.getResourceId(index, i12);
                            if (resourceId10 == -1) {
                                resourceId10 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.r = resourceId10;
                            continue;
                        case 32:
                            int i13 = this.s;
                            int[] iArr11 = d.f1965a;
                            int resourceId11 = obtainStyledAttributes.getResourceId(index, i13);
                            if (resourceId11 == -1) {
                                resourceId11 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.s = resourceId11;
                            continue;
                        case 33:
                            this.G = obtainStyledAttributes.getDimensionPixelSize(index, this.G);
                            continue;
                        case 34:
                            int i14 = this.n;
                            int[] iArr12 = d.f1965a;
                            int resourceId12 = obtainStyledAttributes.getResourceId(index, i14);
                            if (resourceId12 == -1) {
                                resourceId12 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.n = resourceId12;
                            continue;
                        case 35:
                            int i15 = this.m;
                            int[] iArr13 = d.f1965a;
                            int resourceId13 = obtainStyledAttributes.getResourceId(index, i15);
                            if (resourceId13 == -1) {
                                resourceId13 = obtainStyledAttributes.getInt(index, -1);
                            }
                            this.m = resourceId13;
                            continue;
                        case 36:
                            this.w = obtainStyledAttributes.getFloat(index, this.w);
                            continue;
                        case 37:
                            this.R = obtainStyledAttributes.getFloat(index, this.R);
                            continue;
                        case 38:
                            this.Q = obtainStyledAttributes.getFloat(index, this.Q);
                            continue;
                        case 39:
                            this.S = obtainStyledAttributes.getInt(index, this.S);
                            continue;
                        case 40:
                            this.T = obtainStyledAttributes.getInt(index, this.T);
                            continue;
                        default:
                            switch (i2) {
                                case 54:
                                    this.U = obtainStyledAttributes.getInt(index, this.U);
                                    continue;
                                case 55:
                                    this.V = obtainStyledAttributes.getInt(index, this.V);
                                    continue;
                                case 56:
                                    this.W = obtainStyledAttributes.getDimensionPixelSize(index, this.W);
                                    continue;
                                case 57:
                                    this.X = obtainStyledAttributes.getDimensionPixelSize(index, this.X);
                                    continue;
                                case 58:
                                    this.Y = obtainStyledAttributes.getDimensionPixelSize(index, this.Y);
                                    continue;
                                case 59:
                                    this.Z = obtainStyledAttributes.getDimensionPixelSize(index, this.Z);
                                    continue;
                                default:
                                    switch (i2) {
                                        case 61:
                                            int i16 = this.y;
                                            int[] iArr14 = d.f1965a;
                                            int resourceId14 = obtainStyledAttributes.getResourceId(index, i16);
                                            if (resourceId14 == -1) {
                                                resourceId14 = obtainStyledAttributes.getInt(index, -1);
                                            }
                                            this.y = resourceId14;
                                            continue;
                                        case 62:
                                            this.z = obtainStyledAttributes.getDimensionPixelSize(index, this.z);
                                            continue;
                                        case 63:
                                            this.A = obtainStyledAttributes.getFloat(index, this.A);
                                            continue;
                                        default:
                                            switch (i2) {
                                                case 69:
                                                    this.a0 = obtainStyledAttributes.getFloat(index, 1.0f);
                                                    continue;
                                                case 70:
                                                    this.b0 = obtainStyledAttributes.getFloat(index, 1.0f);
                                                    continue;
                                                case 71:
                                                    Log.e("ConstraintSet", "CURRENTLY UNSUPPORTED");
                                                    continue;
                                                case 72:
                                                    this.c0 = obtainStyledAttributes.getInt(index, this.c0);
                                                    continue;
                                                case 73:
                                                    this.d0 = obtainStyledAttributes.getDimensionPixelSize(index, this.d0);
                                                    continue;
                                                case 74:
                                                    this.g0 = obtainStyledAttributes.getString(index);
                                                    continue;
                                                case 75:
                                                    this.k0 = obtainStyledAttributes.getBoolean(index, this.k0);
                                                    continue;
                                                case 76:
                                                    StringBuilder x = c.b.a.a.a.x("unused attribute 0x");
                                                    x.append(Integer.toHexString(index));
                                                    x.append("   ");
                                                    x.append(f1976a.get(index));
                                                    Log.w("ConstraintSet", x.toString());
                                                    continue;
                                                case 77:
                                                    this.h0 = obtainStyledAttributes.getString(index);
                                                    continue;
                                                default:
                                                    StringBuilder x2 = c.b.a.a.a.x("Unknown attribute 0x");
                                                    x2.append(Integer.toHexString(index));
                                                    x2.append("   ");
                                                    x2.append(f1976a.get(index));
                                                    Log.w("ConstraintSet", x2.toString());
                                                    continue;
                                                    continue;
                                                    continue;
                                                    continue;
                                            }
                                    }
                            }
                    }
                } else {
                    this.j0 = obtainStyledAttributes.getBoolean(index, this.j0);
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    /* compiled from: ConstraintSet.java */
    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public static SparseIntArray f1984a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f1985b = false;

        /* renamed from: c  reason: collision with root package name */
        public int f1986c = -1;

        /* renamed from: d  reason: collision with root package name */
        public String f1987d = null;

        /* renamed from: e  reason: collision with root package name */
        public int f1988e = -1;

        /* renamed from: f  reason: collision with root package name */
        public int f1989f = 0;

        /* renamed from: g  reason: collision with root package name */
        public float f1990g = Float.NaN;

        /* renamed from: h  reason: collision with root package name */
        public float f1991h = Float.NaN;

        static {
            SparseIntArray sparseIntArray = new SparseIntArray();
            f1984a = sparseIntArray;
            sparseIntArray.append(2, 1);
            f1984a.append(4, 2);
            f1984a.append(5, 3);
            f1984a.append(1, 4);
            f1984a.append(0, 5);
            f1984a.append(3, 6);
        }

        public void a(Context context, AttributeSet attributeSet) {
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2014f);
            this.f1985b = true;
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                switch (f1984a.get(index)) {
                    case 1:
                        this.f1991h = obtainStyledAttributes.getFloat(index, this.f1991h);
                        break;
                    case 2:
                        this.f1988e = obtainStyledAttributes.getInt(index, this.f1988e);
                        break;
                    case 3:
                        if (obtainStyledAttributes.peekValue(index).type == 3) {
                            this.f1987d = obtainStyledAttributes.getString(index);
                            break;
                        } else {
                            this.f1987d = b.h.a.a.a.f1811a[obtainStyledAttributes.getInteger(index, 0)];
                            break;
                        }
                    case 4:
                        this.f1989f = obtainStyledAttributes.getInt(index, 0);
                        break;
                    case 5:
                        int i2 = this.f1986c;
                        int[] iArr = d.f1965a;
                        int resourceId = obtainStyledAttributes.getResourceId(index, i2);
                        if (resourceId == -1) {
                            resourceId = obtainStyledAttributes.getInt(index, -1);
                        }
                        this.f1986c = resourceId;
                        break;
                    case 6:
                        this.f1990g = obtainStyledAttributes.getFloat(index, this.f1990g);
                        break;
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    /* compiled from: ConstraintSet.java */
    /* renamed from: b.h.c.d$d  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0030d {

        /* renamed from: a  reason: collision with root package name */
        public boolean f1992a = false;

        /* renamed from: b  reason: collision with root package name */
        public int f1993b = 0;

        /* renamed from: c  reason: collision with root package name */
        public int f1994c = 0;

        /* renamed from: d  reason: collision with root package name */
        public float f1995d = 1.0f;

        /* renamed from: e  reason: collision with root package name */
        public float f1996e = Float.NaN;

        public void a(Context context, AttributeSet attributeSet) {
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2015g);
            this.f1992a = true;
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                if (index == 1) {
                    this.f1995d = obtainStyledAttributes.getFloat(index, this.f1995d);
                } else if (index == 0) {
                    int i2 = obtainStyledAttributes.getInt(index, this.f1993b);
                    this.f1993b = i2;
                    int[] iArr = d.f1965a;
                    this.f1993b = d.f1965a[i2];
                } else if (index == 4) {
                    this.f1994c = obtainStyledAttributes.getInt(index, this.f1994c);
                } else if (index == 3) {
                    this.f1996e = obtainStyledAttributes.getFloat(index, this.f1996e);
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    /* compiled from: ConstraintSet.java */
    /* loaded from: classes.dex */
    public static class e {

        /* renamed from: a  reason: collision with root package name */
        public static SparseIntArray f1997a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f1998b = false;

        /* renamed from: c  reason: collision with root package name */
        public float f1999c = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

        /* renamed from: d  reason: collision with root package name */
        public float f2000d = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

        /* renamed from: e  reason: collision with root package name */
        public float f2001e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

        /* renamed from: f  reason: collision with root package name */
        public float f2002f = 1.0f;

        /* renamed from: g  reason: collision with root package name */
        public float f2003g = 1.0f;

        /* renamed from: h  reason: collision with root package name */
        public float f2004h = Float.NaN;
        public float i = Float.NaN;
        public float j = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float k = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public float l = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        public boolean m = false;
        public float n = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

        static {
            SparseIntArray sparseIntArray = new SparseIntArray();
            f1997a = sparseIntArray;
            sparseIntArray.append(6, 1);
            f1997a.append(7, 2);
            f1997a.append(8, 3);
            f1997a.append(4, 4);
            f1997a.append(5, 5);
            f1997a.append(0, 6);
            f1997a.append(1, 7);
            f1997a.append(2, 8);
            f1997a.append(3, 9);
            f1997a.append(9, 10);
            f1997a.append(10, 11);
        }

        public void a(Context context, AttributeSet attributeSet) {
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.i);
            this.f1998b = true;
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i = 0; i < indexCount; i++) {
                int index = obtainStyledAttributes.getIndex(i);
                switch (f1997a.get(index)) {
                    case 1:
                        this.f1999c = obtainStyledAttributes.getFloat(index, this.f1999c);
                        break;
                    case 2:
                        this.f2000d = obtainStyledAttributes.getFloat(index, this.f2000d);
                        break;
                    case 3:
                        this.f2001e = obtainStyledAttributes.getFloat(index, this.f2001e);
                        break;
                    case 4:
                        this.f2002f = obtainStyledAttributes.getFloat(index, this.f2002f);
                        break;
                    case 5:
                        this.f2003g = obtainStyledAttributes.getFloat(index, this.f2003g);
                        break;
                    case 6:
                        this.f2004h = obtainStyledAttributes.getDimension(index, this.f2004h);
                        break;
                    case 7:
                        this.i = obtainStyledAttributes.getDimension(index, this.i);
                        break;
                    case 8:
                        this.j = obtainStyledAttributes.getDimension(index, this.j);
                        break;
                    case 9:
                        this.k = obtainStyledAttributes.getDimension(index, this.k);
                        break;
                    case 10:
                        this.l = obtainStyledAttributes.getDimension(index, this.l);
                        break;
                    case 11:
                        this.m = true;
                        this.n = obtainStyledAttributes.getDimension(index, this.n);
                        break;
                }
            }
            obtainStyledAttributes.recycle();
        }
    }

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        f1966b = sparseIntArray;
        sparseIntArray.append(77, 25);
        f1966b.append(78, 26);
        f1966b.append(80, 29);
        f1966b.append(81, 30);
        f1966b.append(87, 36);
        f1966b.append(86, 35);
        f1966b.append(59, 4);
        f1966b.append(58, 3);
        f1966b.append(56, 1);
        f1966b.append(95, 6);
        f1966b.append(96, 7);
        f1966b.append(66, 17);
        f1966b.append(67, 18);
        f1966b.append(68, 19);
        f1966b.append(0, 27);
        f1966b.append(82, 32);
        f1966b.append(83, 33);
        f1966b.append(65, 10);
        f1966b.append(64, 9);
        f1966b.append(99, 13);
        f1966b.append(102, 16);
        f1966b.append(100, 14);
        f1966b.append(97, 11);
        f1966b.append(101, 15);
        f1966b.append(98, 12);
        f1966b.append(90, 40);
        f1966b.append(75, 39);
        f1966b.append(74, 41);
        f1966b.append(89, 42);
        f1966b.append(73, 20);
        f1966b.append(88, 37);
        f1966b.append(63, 5);
        f1966b.append(76, 82);
        f1966b.append(85, 82);
        f1966b.append(79, 82);
        f1966b.append(57, 82);
        f1966b.append(55, 82);
        f1966b.append(5, 24);
        f1966b.append(7, 28);
        f1966b.append(23, 31);
        f1966b.append(24, 8);
        f1966b.append(6, 34);
        f1966b.append(8, 2);
        f1966b.append(3, 23);
        f1966b.append(4, 21);
        f1966b.append(2, 22);
        f1966b.append(13, 43);
        f1966b.append(26, 44);
        f1966b.append(21, 45);
        f1966b.append(22, 46);
        f1966b.append(20, 60);
        f1966b.append(18, 47);
        f1966b.append(19, 48);
        f1966b.append(14, 49);
        f1966b.append(15, 50);
        f1966b.append(16, 51);
        f1966b.append(17, 52);
        f1966b.append(25, 53);
        f1966b.append(91, 54);
        f1966b.append(69, 55);
        f1966b.append(92, 56);
        f1966b.append(70, 57);
        f1966b.append(93, 58);
        f1966b.append(71, 59);
        f1966b.append(60, 61);
        f1966b.append(62, 62);
        f1966b.append(61, 63);
        f1966b.append(27, 64);
        f1966b.append(107, 65);
        f1966b.append(34, 66);
        f1966b.append(108, 67);
        f1966b.append(104, 79);
        f1966b.append(1, 38);
        f1966b.append(103, 68);
        f1966b.append(94, 69);
        f1966b.append(72, 70);
        f1966b.append(31, 71);
        f1966b.append(29, 72);
        f1966b.append(30, 73);
        f1966b.append(32, 74);
        f1966b.append(28, 75);
        f1966b.append(105, 76);
        f1966b.append(84, 77);
        f1966b.append(109, 78);
        f1966b.append(54, 80);
        f1966b.append(53, 81);
    }

    public void a(ConstraintLayout constraintLayout) {
        b(constraintLayout, true);
        constraintLayout.setConstraintSet(null);
        constraintLayout.requestLayout();
    }

    public void b(ConstraintLayout constraintLayout, boolean z) {
        int i;
        Iterator<String> it;
        String str;
        int childCount = constraintLayout.getChildCount();
        HashSet hashSet = new HashSet(this.f1969e.keySet());
        int i2 = 0;
        while (i2 < childCount) {
            View childAt = constraintLayout.getChildAt(i2);
            int id = childAt.getId();
            if (!this.f1969e.containsKey(Integer.valueOf(id))) {
                StringBuilder x = c.b.a.a.a.x("id unknown ");
                try {
                    str = childAt.getContext().getResources().getResourceEntryName(childAt.getId());
                } catch (Exception unused) {
                    str = "UNKNOWN";
                }
                x.append(str);
                Log.w("ConstraintSet", x.toString());
            } else if (this.f1968d && id == -1) {
                throw new RuntimeException("All children of ConstraintLayout must have ids to use ConstraintSet");
            } else {
                if (id != -1) {
                    if (this.f1969e.containsKey(Integer.valueOf(id))) {
                        hashSet.remove(Integer.valueOf(id));
                        a aVar = this.f1969e.get(Integer.valueOf(id));
                        if (childAt instanceof Barrier) {
                            aVar.f1973d.e0 = 1;
                        }
                        int i3 = aVar.f1973d.e0;
                        if (i3 != -1 && i3 == 1) {
                            Barrier barrier = (Barrier) childAt;
                            barrier.setId(id);
                            barrier.setType(aVar.f1973d.c0);
                            barrier.setMargin(aVar.f1973d.d0);
                            barrier.setAllowsGoneWidget(aVar.f1973d.k0);
                            b bVar = aVar.f1973d;
                            int[] iArr = bVar.f0;
                            if (iArr != null) {
                                barrier.setReferencedIds(iArr);
                            } else {
                                String str2 = bVar.g0;
                                if (str2 != null) {
                                    bVar.f0 = d(barrier, str2);
                                    barrier.setReferencedIds(aVar.f1973d.f0);
                                }
                            }
                        }
                        ConstraintLayout.a aVar2 = (ConstraintLayout.a) childAt.getLayoutParams();
                        aVar2.a();
                        aVar.a(aVar2);
                        if (z) {
                            HashMap<String, b.h.c.a> hashMap = aVar.f1975f;
                            Class<?> cls = childAt.getClass();
                            Iterator<String> it2 = hashMap.keySet().iterator();
                            while (it2.hasNext()) {
                                String next = it2.next();
                                b.h.c.a aVar3 = hashMap.get(next);
                                int i4 = childCount;
                                String q = c.b.a.a.a.q("set", next);
                                HashMap<String, b.h.c.a> hashMap2 = hashMap;
                                try {
                                    switch (m0.f(aVar3.f1937b)) {
                                        case 0:
                                            it = it2;
                                            cls.getMethod(q, Integer.TYPE).invoke(childAt, Integer.valueOf(aVar3.f1938c));
                                            break;
                                        case 1:
                                            it = it2;
                                            cls.getMethod(q, Float.TYPE).invoke(childAt, Float.valueOf(aVar3.f1939d));
                                            break;
                                        case 2:
                                            it = it2;
                                            cls.getMethod(q, Integer.TYPE).invoke(childAt, Integer.valueOf(aVar3.f1942g));
                                            break;
                                        case 3:
                                            it = it2;
                                            Method method = cls.getMethod(q, Drawable.class);
                                            ColorDrawable colorDrawable = new ColorDrawable();
                                            colorDrawable.setColor(aVar3.f1942g);
                                            method.invoke(childAt, colorDrawable);
                                            break;
                                        case 4:
                                            it = it2;
                                            cls.getMethod(q, CharSequence.class).invoke(childAt, aVar3.f1940e);
                                            break;
                                        case 5:
                                            it = it2;
                                            cls.getMethod(q, Boolean.TYPE).invoke(childAt, Boolean.valueOf(aVar3.f1941f));
                                            break;
                                        case 6:
                                            it = it2;
                                            try {
                                                cls.getMethod(q, Float.TYPE).invoke(childAt, Float.valueOf(aVar3.f1939d));
                                            } catch (IllegalAccessException e2) {
                                                e = e2;
                                                StringBuilder B = c.b.a.a.a.B(" Custom Attribute \"", next, "\" not found on ");
                                                B.append(cls.getName());
                                                Log.e("TransitionLayout", B.toString());
                                                e.printStackTrace();
                                                childCount = i4;
                                                hashMap = hashMap2;
                                                it2 = it;
                                            } catch (NoSuchMethodException e3) {
                                                e = e3;
                                                Log.e("TransitionLayout", e.getMessage());
                                                Log.e("TransitionLayout", " Custom Attribute \"" + next + "\" not found on " + cls.getName());
                                                StringBuilder sb = new StringBuilder();
                                                sb.append(cls.getName());
                                                sb.append(" must have a method ");
                                                sb.append(q);
                                                Log.e("TransitionLayout", sb.toString());
                                                childCount = i4;
                                                hashMap = hashMap2;
                                                it2 = it;
                                            } catch (InvocationTargetException e4) {
                                                e = e4;
                                                StringBuilder B2 = c.b.a.a.a.B(" Custom Attribute \"", next, "\" not found on ");
                                                B2.append(cls.getName());
                                                Log.e("TransitionLayout", B2.toString());
                                                e.printStackTrace();
                                                childCount = i4;
                                                hashMap = hashMap2;
                                                it2 = it;
                                            }
                                        default:
                                            it = it2;
                                            break;
                                    }
                                } catch (IllegalAccessException e5) {
                                    e = e5;
                                    it = it2;
                                } catch (NoSuchMethodException e6) {
                                    e = e6;
                                    it = it2;
                                } catch (InvocationTargetException e7) {
                                    e = e7;
                                    it = it2;
                                }
                                childCount = i4;
                                hashMap = hashMap2;
                                it2 = it;
                            }
                        }
                        i = childCount;
                        childAt.setLayoutParams(aVar2);
                        C0030d c0030d = aVar.f1971b;
                        if (c0030d.f1994c == 0) {
                            childAt.setVisibility(c0030d.f1993b);
                        }
                        childAt.setAlpha(aVar.f1971b.f1995d);
                        childAt.setRotation(aVar.f1974e.f1999c);
                        childAt.setRotationX(aVar.f1974e.f2000d);
                        childAt.setRotationY(aVar.f1974e.f2001e);
                        childAt.setScaleX(aVar.f1974e.f2002f);
                        childAt.setScaleY(aVar.f1974e.f2003g);
                        if (!Float.isNaN(aVar.f1974e.f2004h)) {
                            childAt.setPivotX(aVar.f1974e.f2004h);
                        }
                        if (!Float.isNaN(aVar.f1974e.i)) {
                            childAt.setPivotY(aVar.f1974e.i);
                        }
                        childAt.setTranslationX(aVar.f1974e.j);
                        childAt.setTranslationY(aVar.f1974e.k);
                        childAt.setTranslationZ(aVar.f1974e.l);
                        e eVar = aVar.f1974e;
                        if (eVar.m) {
                            childAt.setElevation(eVar.n);
                        }
                    } else {
                        i = childCount;
                        Log.v("ConstraintSet", "WARNING NO CONSTRAINTS for view " + id);
                    }
                    i2++;
                    childCount = i;
                }
            }
            i = childCount;
            i2++;
            childCount = i;
        }
        Iterator it3 = hashSet.iterator();
        while (it3.hasNext()) {
            Integer num = (Integer) it3.next();
            a aVar4 = this.f1969e.get(num);
            int i5 = aVar4.f1973d.e0;
            if (i5 != -1 && i5 == 1) {
                Barrier barrier2 = new Barrier(constraintLayout.getContext());
                barrier2.setId(num.intValue());
                b bVar2 = aVar4.f1973d;
                int[] iArr2 = bVar2.f0;
                if (iArr2 != null) {
                    barrier2.setReferencedIds(iArr2);
                } else {
                    String str3 = bVar2.g0;
                    if (str3 != null) {
                        bVar2.f0 = d(barrier2, str3);
                        barrier2.setReferencedIds(aVar4.f1973d.f0);
                    }
                }
                barrier2.setType(aVar4.f1973d.c0);
                barrier2.setMargin(aVar4.f1973d.d0);
                ConstraintLayout.a generateDefaultLayoutParams = constraintLayout.generateDefaultLayoutParams();
                barrier2.k();
                aVar4.a(generateDefaultLayoutParams);
                constraintLayout.addView(barrier2, generateDefaultLayoutParams);
            }
            if (aVar4.f1973d.f1977b) {
                View guideline = new Guideline(constraintLayout.getContext());
                guideline.setId(num.intValue());
                ConstraintLayout.a generateDefaultLayoutParams2 = constraintLayout.generateDefaultLayoutParams();
                aVar4.a(generateDefaultLayoutParams2);
                constraintLayout.addView(guideline, generateDefaultLayoutParams2);
            }
        }
    }

    public void c(ConstraintLayout constraintLayout) {
        d dVar = this;
        int childCount = constraintLayout.getChildCount();
        dVar.f1969e.clear();
        int i = 0;
        while (i < childCount) {
            View childAt = constraintLayout.getChildAt(i);
            ConstraintLayout.a aVar = (ConstraintLayout.a) childAt.getLayoutParams();
            int id = childAt.getId();
            if (dVar.f1968d && id == -1) {
                throw new RuntimeException("All children of ConstraintLayout must have ids to use ConstraintSet");
            }
            if (!dVar.f1969e.containsKey(Integer.valueOf(id))) {
                dVar.f1969e.put(Integer.valueOf(id), new a());
            }
            a aVar2 = dVar.f1969e.get(Integer.valueOf(id));
            HashMap<String, b.h.c.a> hashMap = dVar.f1967c;
            HashMap<String, b.h.c.a> hashMap2 = new HashMap<>();
            Class<?> cls = childAt.getClass();
            for (String str : hashMap.keySet()) {
                b.h.c.a aVar3 = hashMap.get(str);
                try {
                } catch (IllegalAccessException e2) {
                    e = e2;
                } catch (NoSuchMethodException e3) {
                    e = e3;
                } catch (InvocationTargetException e4) {
                    e = e4;
                }
                if (str.equals("BackgroundColor")) {
                    hashMap2.put(str, new b.h.c.a(aVar3, Integer.valueOf(((ColorDrawable) childAt.getBackground()).getColor())));
                } else {
                    try {
                        hashMap2.put(str, new b.h.c.a(aVar3, cls.getMethod("getMap" + str, new Class[0]).invoke(childAt, new Object[0])));
                    } catch (IllegalAccessException e5) {
                        e = e5;
                        e.printStackTrace();
                    } catch (NoSuchMethodException e6) {
                        e = e6;
                        e.printStackTrace();
                    } catch (InvocationTargetException e7) {
                        e = e7;
                        e.printStackTrace();
                    }
                }
            }
            aVar2.f1975f = hashMap2;
            aVar2.b(id, aVar);
            aVar2.f1971b.f1993b = childAt.getVisibility();
            aVar2.f1971b.f1995d = childAt.getAlpha();
            aVar2.f1974e.f1999c = childAt.getRotation();
            aVar2.f1974e.f2000d = childAt.getRotationX();
            aVar2.f1974e.f2001e = childAt.getRotationY();
            aVar2.f1974e.f2002f = childAt.getScaleX();
            aVar2.f1974e.f2003g = childAt.getScaleY();
            float pivotX = childAt.getPivotX();
            float pivotY = childAt.getPivotY();
            if (pivotX != ShadowDrawableWrapper.COS_45 || pivotY != ShadowDrawableWrapper.COS_45) {
                e eVar = aVar2.f1974e;
                eVar.f2004h = pivotX;
                eVar.i = pivotY;
            }
            aVar2.f1974e.j = childAt.getTranslationX();
            aVar2.f1974e.k = childAt.getTranslationY();
            aVar2.f1974e.l = childAt.getTranslationZ();
            e eVar2 = aVar2.f1974e;
            if (eVar2.m) {
                eVar2.n = childAt.getElevation();
            }
            if (childAt instanceof Barrier) {
                Barrier barrier = (Barrier) childAt;
                b bVar = aVar2.f1973d;
                bVar.k0 = barrier.k.o0;
                bVar.f0 = barrier.getReferencedIds();
                aVar2.f1973d.c0 = barrier.getType();
                aVar2.f1973d.d0 = barrier.getMargin();
            }
            i++;
            dVar = this;
        }
    }

    public final int[] d(View view, String str) {
        int i;
        Object designInformation;
        String[] split = str.split(",");
        Context context = view.getContext();
        int[] iArr = new int[split.length];
        int i2 = 0;
        int i3 = 0;
        while (i2 < split.length) {
            String trim = split[i2].trim();
            try {
                i = h.class.getField(trim).getInt(null);
            } catch (Exception unused) {
                i = 0;
            }
            if (i == 0) {
                i = context.getResources().getIdentifier(trim, "id", context.getPackageName());
            }
            if (i == 0 && view.isInEditMode() && (view.getParent() instanceof ConstraintLayout) && (designInformation = ((ConstraintLayout) view.getParent()).getDesignInformation(0, trim)) != null && (designInformation instanceof Integer)) {
                i = ((Integer) designInformation).intValue();
            }
            iArr[i3] = i;
            i2++;
            i3++;
        }
        return i3 != split.length ? Arrays.copyOf(iArr, i3) : iArr;
    }

    public final a e(Context context, AttributeSet attributeSet) {
        a aVar = new a();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2009a);
        int indexCount = obtainStyledAttributes.getIndexCount();
        for (int i = 0; i < indexCount; i++) {
            int index = obtainStyledAttributes.getIndex(i);
            if (index != 1 && 23 != index && 24 != index) {
                aVar.f1972c.f1985b = true;
                aVar.f1973d.f1978c = true;
                aVar.f1971b.f1992a = true;
                aVar.f1974e.f1998b = true;
            }
            switch (f1966b.get(index)) {
                case 1:
                    b bVar = aVar.f1973d;
                    int resourceId = obtainStyledAttributes.getResourceId(index, bVar.q);
                    if (resourceId == -1) {
                        resourceId = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar.q = resourceId;
                    break;
                case 2:
                    b bVar2 = aVar.f1973d;
                    bVar2.H = obtainStyledAttributes.getDimensionPixelSize(index, bVar2.H);
                    break;
                case 3:
                    b bVar3 = aVar.f1973d;
                    int resourceId2 = obtainStyledAttributes.getResourceId(index, bVar3.p);
                    if (resourceId2 == -1) {
                        resourceId2 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar3.p = resourceId2;
                    break;
                case 4:
                    b bVar4 = aVar.f1973d;
                    int resourceId3 = obtainStyledAttributes.getResourceId(index, bVar4.o);
                    if (resourceId3 == -1) {
                        resourceId3 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar4.o = resourceId3;
                    break;
                case 5:
                    aVar.f1973d.x = obtainStyledAttributes.getString(index);
                    break;
                case 6:
                    b bVar5 = aVar.f1973d;
                    bVar5.B = obtainStyledAttributes.getDimensionPixelOffset(index, bVar5.B);
                    break;
                case 7:
                    b bVar6 = aVar.f1973d;
                    bVar6.C = obtainStyledAttributes.getDimensionPixelOffset(index, bVar6.C);
                    break;
                case 8:
                    b bVar7 = aVar.f1973d;
                    bVar7.I = obtainStyledAttributes.getDimensionPixelSize(index, bVar7.I);
                    break;
                case 9:
                    b bVar8 = aVar.f1973d;
                    int resourceId4 = obtainStyledAttributes.getResourceId(index, bVar8.u);
                    if (resourceId4 == -1) {
                        resourceId4 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar8.u = resourceId4;
                    break;
                case 10:
                    b bVar9 = aVar.f1973d;
                    int resourceId5 = obtainStyledAttributes.getResourceId(index, bVar9.t);
                    if (resourceId5 == -1) {
                        resourceId5 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar9.t = resourceId5;
                    break;
                case 11:
                    b bVar10 = aVar.f1973d;
                    bVar10.N = obtainStyledAttributes.getDimensionPixelSize(index, bVar10.N);
                    break;
                case 12:
                    b bVar11 = aVar.f1973d;
                    bVar11.O = obtainStyledAttributes.getDimensionPixelSize(index, bVar11.O);
                    break;
                case 13:
                    b bVar12 = aVar.f1973d;
                    bVar12.K = obtainStyledAttributes.getDimensionPixelSize(index, bVar12.K);
                    break;
                case 14:
                    b bVar13 = aVar.f1973d;
                    bVar13.M = obtainStyledAttributes.getDimensionPixelSize(index, bVar13.M);
                    break;
                case 15:
                    b bVar14 = aVar.f1973d;
                    bVar14.P = obtainStyledAttributes.getDimensionPixelSize(index, bVar14.P);
                    break;
                case 16:
                    b bVar15 = aVar.f1973d;
                    bVar15.L = obtainStyledAttributes.getDimensionPixelSize(index, bVar15.L);
                    break;
                case 17:
                    b bVar16 = aVar.f1973d;
                    bVar16.f1981f = obtainStyledAttributes.getDimensionPixelOffset(index, bVar16.f1981f);
                    break;
                case 18:
                    b bVar17 = aVar.f1973d;
                    bVar17.f1982g = obtainStyledAttributes.getDimensionPixelOffset(index, bVar17.f1982g);
                    break;
                case 19:
                    b bVar18 = aVar.f1973d;
                    bVar18.f1983h = obtainStyledAttributes.getFloat(index, bVar18.f1983h);
                    break;
                case 20:
                    b bVar19 = aVar.f1973d;
                    bVar19.v = obtainStyledAttributes.getFloat(index, bVar19.v);
                    break;
                case 21:
                    b bVar20 = aVar.f1973d;
                    bVar20.f1980e = obtainStyledAttributes.getLayoutDimension(index, bVar20.f1980e);
                    break;
                case 22:
                    C0030d c0030d = aVar.f1971b;
                    c0030d.f1993b = obtainStyledAttributes.getInt(index, c0030d.f1993b);
                    C0030d c0030d2 = aVar.f1971b;
                    c0030d2.f1993b = f1965a[c0030d2.f1993b];
                    break;
                case 23:
                    b bVar21 = aVar.f1973d;
                    bVar21.f1979d = obtainStyledAttributes.getLayoutDimension(index, bVar21.f1979d);
                    break;
                case 24:
                    b bVar22 = aVar.f1973d;
                    bVar22.E = obtainStyledAttributes.getDimensionPixelSize(index, bVar22.E);
                    break;
                case 25:
                    b bVar23 = aVar.f1973d;
                    int resourceId6 = obtainStyledAttributes.getResourceId(index, bVar23.i);
                    if (resourceId6 == -1) {
                        resourceId6 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar23.i = resourceId6;
                    break;
                case 26:
                    b bVar24 = aVar.f1973d;
                    int resourceId7 = obtainStyledAttributes.getResourceId(index, bVar24.j);
                    if (resourceId7 == -1) {
                        resourceId7 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar24.j = resourceId7;
                    break;
                case 27:
                    b bVar25 = aVar.f1973d;
                    bVar25.D = obtainStyledAttributes.getInt(index, bVar25.D);
                    break;
                case 28:
                    b bVar26 = aVar.f1973d;
                    bVar26.F = obtainStyledAttributes.getDimensionPixelSize(index, bVar26.F);
                    break;
                case 29:
                    b bVar27 = aVar.f1973d;
                    int resourceId8 = obtainStyledAttributes.getResourceId(index, bVar27.k);
                    if (resourceId8 == -1) {
                        resourceId8 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar27.k = resourceId8;
                    break;
                case 30:
                    b bVar28 = aVar.f1973d;
                    int resourceId9 = obtainStyledAttributes.getResourceId(index, bVar28.l);
                    if (resourceId9 == -1) {
                        resourceId9 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar28.l = resourceId9;
                    break;
                case 31:
                    b bVar29 = aVar.f1973d;
                    bVar29.J = obtainStyledAttributes.getDimensionPixelSize(index, bVar29.J);
                    break;
                case 32:
                    b bVar30 = aVar.f1973d;
                    int resourceId10 = obtainStyledAttributes.getResourceId(index, bVar30.r);
                    if (resourceId10 == -1) {
                        resourceId10 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar30.r = resourceId10;
                    break;
                case 33:
                    b bVar31 = aVar.f1973d;
                    int resourceId11 = obtainStyledAttributes.getResourceId(index, bVar31.s);
                    if (resourceId11 == -1) {
                        resourceId11 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar31.s = resourceId11;
                    break;
                case 34:
                    b bVar32 = aVar.f1973d;
                    bVar32.G = obtainStyledAttributes.getDimensionPixelSize(index, bVar32.G);
                    break;
                case 35:
                    b bVar33 = aVar.f1973d;
                    int resourceId12 = obtainStyledAttributes.getResourceId(index, bVar33.n);
                    if (resourceId12 == -1) {
                        resourceId12 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar33.n = resourceId12;
                    break;
                case 36:
                    b bVar34 = aVar.f1973d;
                    int resourceId13 = obtainStyledAttributes.getResourceId(index, bVar34.m);
                    if (resourceId13 == -1) {
                        resourceId13 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar34.m = resourceId13;
                    break;
                case 37:
                    b bVar35 = aVar.f1973d;
                    bVar35.w = obtainStyledAttributes.getFloat(index, bVar35.w);
                    break;
                case 38:
                    aVar.f1970a = obtainStyledAttributes.getResourceId(index, aVar.f1970a);
                    break;
                case 39:
                    b bVar36 = aVar.f1973d;
                    bVar36.R = obtainStyledAttributes.getFloat(index, bVar36.R);
                    break;
                case 40:
                    b bVar37 = aVar.f1973d;
                    bVar37.Q = obtainStyledAttributes.getFloat(index, bVar37.Q);
                    break;
                case 41:
                    b bVar38 = aVar.f1973d;
                    bVar38.S = obtainStyledAttributes.getInt(index, bVar38.S);
                    break;
                case 42:
                    b bVar39 = aVar.f1973d;
                    bVar39.T = obtainStyledAttributes.getInt(index, bVar39.T);
                    break;
                case 43:
                    C0030d c0030d3 = aVar.f1971b;
                    c0030d3.f1995d = obtainStyledAttributes.getFloat(index, c0030d3.f1995d);
                    break;
                case 44:
                    e eVar = aVar.f1974e;
                    eVar.m = true;
                    eVar.n = obtainStyledAttributes.getDimension(index, eVar.n);
                    break;
                case 45:
                    e eVar2 = aVar.f1974e;
                    eVar2.f2000d = obtainStyledAttributes.getFloat(index, eVar2.f2000d);
                    break;
                case 46:
                    e eVar3 = aVar.f1974e;
                    eVar3.f2001e = obtainStyledAttributes.getFloat(index, eVar3.f2001e);
                    break;
                case 47:
                    e eVar4 = aVar.f1974e;
                    eVar4.f2002f = obtainStyledAttributes.getFloat(index, eVar4.f2002f);
                    break;
                case 48:
                    e eVar5 = aVar.f1974e;
                    eVar5.f2003g = obtainStyledAttributes.getFloat(index, eVar5.f2003g);
                    break;
                case 49:
                    e eVar6 = aVar.f1974e;
                    eVar6.f2004h = obtainStyledAttributes.getDimension(index, eVar6.f2004h);
                    break;
                case 50:
                    e eVar7 = aVar.f1974e;
                    eVar7.i = obtainStyledAttributes.getDimension(index, eVar7.i);
                    break;
                case 51:
                    e eVar8 = aVar.f1974e;
                    eVar8.j = obtainStyledAttributes.getDimension(index, eVar8.j);
                    break;
                case 52:
                    e eVar9 = aVar.f1974e;
                    eVar9.k = obtainStyledAttributes.getDimension(index, eVar9.k);
                    break;
                case 53:
                    e eVar10 = aVar.f1974e;
                    eVar10.l = obtainStyledAttributes.getDimension(index, eVar10.l);
                    break;
                case 54:
                    b bVar40 = aVar.f1973d;
                    bVar40.U = obtainStyledAttributes.getInt(index, bVar40.U);
                    break;
                case 55:
                    b bVar41 = aVar.f1973d;
                    bVar41.V = obtainStyledAttributes.getInt(index, bVar41.V);
                    break;
                case 56:
                    b bVar42 = aVar.f1973d;
                    bVar42.W = obtainStyledAttributes.getDimensionPixelSize(index, bVar42.W);
                    break;
                case 57:
                    b bVar43 = aVar.f1973d;
                    bVar43.X = obtainStyledAttributes.getDimensionPixelSize(index, bVar43.X);
                    break;
                case 58:
                    b bVar44 = aVar.f1973d;
                    bVar44.Y = obtainStyledAttributes.getDimensionPixelSize(index, bVar44.Y);
                    break;
                case 59:
                    b bVar45 = aVar.f1973d;
                    bVar45.Z = obtainStyledAttributes.getDimensionPixelSize(index, bVar45.Z);
                    break;
                case 60:
                    e eVar11 = aVar.f1974e;
                    eVar11.f1999c = obtainStyledAttributes.getFloat(index, eVar11.f1999c);
                    break;
                case 61:
                    b bVar46 = aVar.f1973d;
                    int resourceId14 = obtainStyledAttributes.getResourceId(index, bVar46.y);
                    if (resourceId14 == -1) {
                        resourceId14 = obtainStyledAttributes.getInt(index, -1);
                    }
                    bVar46.y = resourceId14;
                    break;
                case 62:
                    b bVar47 = aVar.f1973d;
                    bVar47.z = obtainStyledAttributes.getDimensionPixelSize(index, bVar47.z);
                    break;
                case 63:
                    b bVar48 = aVar.f1973d;
                    bVar48.A = obtainStyledAttributes.getFloat(index, bVar48.A);
                    break;
                case 64:
                    c cVar = aVar.f1972c;
                    int resourceId15 = obtainStyledAttributes.getResourceId(index, cVar.f1986c);
                    if (resourceId15 == -1) {
                        resourceId15 = obtainStyledAttributes.getInt(index, -1);
                    }
                    cVar.f1986c = resourceId15;
                    break;
                case 65:
                    if (obtainStyledAttributes.peekValue(index).type == 3) {
                        aVar.f1972c.f1987d = obtainStyledAttributes.getString(index);
                        break;
                    } else {
                        aVar.f1972c.f1987d = b.h.a.a.a.f1811a[obtainStyledAttributes.getInteger(index, 0)];
                        break;
                    }
                case 66:
                    aVar.f1972c.f1989f = obtainStyledAttributes.getInt(index, 0);
                    break;
                case 67:
                    c cVar2 = aVar.f1972c;
                    cVar2.f1991h = obtainStyledAttributes.getFloat(index, cVar2.f1991h);
                    break;
                case 68:
                    C0030d c0030d4 = aVar.f1971b;
                    c0030d4.f1996e = obtainStyledAttributes.getFloat(index, c0030d4.f1996e);
                    break;
                case 69:
                    aVar.f1973d.a0 = obtainStyledAttributes.getFloat(index, 1.0f);
                    break;
                case 70:
                    aVar.f1973d.b0 = obtainStyledAttributes.getFloat(index, 1.0f);
                    break;
                case 71:
                    Log.e("ConstraintSet", "CURRENTLY UNSUPPORTED");
                    break;
                case 72:
                    b bVar49 = aVar.f1973d;
                    bVar49.c0 = obtainStyledAttributes.getInt(index, bVar49.c0);
                    break;
                case 73:
                    b bVar50 = aVar.f1973d;
                    bVar50.d0 = obtainStyledAttributes.getDimensionPixelSize(index, bVar50.d0);
                    break;
                case 74:
                    aVar.f1973d.g0 = obtainStyledAttributes.getString(index);
                    break;
                case 75:
                    b bVar51 = aVar.f1973d;
                    bVar51.k0 = obtainStyledAttributes.getBoolean(index, bVar51.k0);
                    break;
                case 76:
                    c cVar3 = aVar.f1972c;
                    cVar3.f1988e = obtainStyledAttributes.getInt(index, cVar3.f1988e);
                    break;
                case 77:
                    aVar.f1973d.h0 = obtainStyledAttributes.getString(index);
                    break;
                case 78:
                    C0030d c0030d5 = aVar.f1971b;
                    c0030d5.f1994c = obtainStyledAttributes.getInt(index, c0030d5.f1994c);
                    break;
                case 79:
                    c cVar4 = aVar.f1972c;
                    cVar4.f1990g = obtainStyledAttributes.getFloat(index, cVar4.f1990g);
                    break;
                case 80:
                    b bVar52 = aVar.f1973d;
                    bVar52.i0 = obtainStyledAttributes.getBoolean(index, bVar52.i0);
                    break;
                case 81:
                    b bVar53 = aVar.f1973d;
                    bVar53.j0 = obtainStyledAttributes.getBoolean(index, bVar53.j0);
                    break;
                case 82:
                    StringBuilder x = c.b.a.a.a.x("unused attribute 0x");
                    x.append(Integer.toHexString(index));
                    x.append("   ");
                    x.append(f1966b.get(index));
                    Log.w("ConstraintSet", x.toString());
                    break;
                default:
                    StringBuilder x2 = c.b.a.a.a.x("Unknown attribute 0x");
                    x2.append(Integer.toHexString(index));
                    x2.append("   ");
                    x2.append(f1966b.get(index));
                    Log.w("ConstraintSet", x2.toString());
                    break;
            }
        }
        obtainStyledAttributes.recycle();
        return aVar;
    }

    public void f(Context context, int i) {
        XmlResourceParser xml = context.getResources().getXml(i);
        try {
            for (int eventType = xml.getEventType(); eventType != 1; eventType = xml.next()) {
                if (eventType == 0) {
                    xml.getName();
                    continue;
                } else if (eventType != 2) {
                    continue;
                } else {
                    String name = xml.getName();
                    a e2 = e(context, Xml.asAttributeSet(xml));
                    if (name.equalsIgnoreCase("Guideline")) {
                        e2.f1973d.f1977b = true;
                    }
                    this.f1969e.put(Integer.valueOf(e2.f1970a), e2);
                    continue;
                }
            }
        } catch (IOException e3) {
            e3.printStackTrace();
        } catch (XmlPullParserException e4) {
            e4.printStackTrace();
        }
    }
}