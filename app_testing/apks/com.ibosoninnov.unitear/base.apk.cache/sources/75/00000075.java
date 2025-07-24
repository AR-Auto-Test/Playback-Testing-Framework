package androidx.constraintlayout.widget;

import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseArray;
import android.util.SparseIntArray;
import android.view.View;
import android.view.ViewGroup;
import b.d.b.m0;
import b.h.b.i.c;
import b.h.b.i.e;
import b.h.b.i.h;
import b.h.b.i.i;
import b.h.b.i.k;
import b.h.b.i.l.b;
import b.h.b.i.l.m;
import b.h.b.i.l.o;
import b.h.c.c;
import b.h.c.d;
import b.h.c.f;
import b.h.c.g;
import b.h.c.j;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Objects;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgcodecs.Imgcodecs;

/* loaded from: classes.dex */
public class ConstraintLayout extends ViewGroup {
    private static final boolean DEBUG = false;
    private static final boolean DEBUG_DRAW_CONSTRAINTS = false;
    public static final int DESIGN_INFO_ID = 0;
    private static final boolean MEASURE = false;
    private static final String TAG = "ConstraintLayout";
    private static final boolean USE_CONSTRAINTS_HELPER = true;
    public static final String VERSION = "ConstraintLayout-2.0.4";
    public SparseArray<View> mChildrenByIds;
    private ArrayList<b.h.c.b> mConstraintHelpers;
    public c mConstraintLayoutSpec;
    private d mConstraintSet;
    private int mConstraintSetId;
    private f mConstraintsChangedListener;
    private HashMap<String, Integer> mDesignIds;
    public boolean mDirtyHierarchy;
    private int mLastMeasureHeight;
    public int mLastMeasureHeightMode;
    public int mLastMeasureHeightSize;
    private int mLastMeasureWidth;
    public int mLastMeasureWidthMode;
    public int mLastMeasureWidthSize;
    public e mLayoutWidget;
    private int mMaxHeight;
    private int mMaxWidth;
    public b mMeasurer;
    private b.h.b.e mMetrics;
    private int mMinHeight;
    private int mMinWidth;
    private int mOnMeasureHeightMeasureSpec;
    private int mOnMeasureWidthMeasureSpec;
    private int mOptimizationLevel;
    private SparseArray<b.h.b.i.d> mTempMapIdToWidget;

    /* loaded from: classes.dex */
    public class b implements b.InterfaceC0029b {

        /* renamed from: a  reason: collision with root package name */
        public ConstraintLayout f206a;

        /* renamed from: b  reason: collision with root package name */
        public int f207b;

        /* renamed from: c  reason: collision with root package name */
        public int f208c;

        /* renamed from: d  reason: collision with root package name */
        public int f209d;

        /* renamed from: e  reason: collision with root package name */
        public int f210e;

        /* renamed from: f  reason: collision with root package name */
        public int f211f;

        /* renamed from: g  reason: collision with root package name */
        public int f212g;

        public b(ConstraintLayout constraintLayout) {
            this.f206a = constraintLayout;
        }

        public final boolean a(int i, int i2, int i3) {
            if (i == i2) {
                return true;
            }
            int mode = View.MeasureSpec.getMode(i);
            View.MeasureSpec.getSize(i);
            int mode2 = View.MeasureSpec.getMode(i2);
            int size = View.MeasureSpec.getSize(i2);
            if (mode2 == 1073741824) {
                return (mode == Integer.MIN_VALUE || mode == 0) && i3 == size;
            }
            return false;
        }

        /* JADX WARN: Removed duplicated region for block: B:124:0x01a5  */
        /* JADX WARN: Removed duplicated region for block: B:127:0x01b8  */
        /* JADX WARN: Removed duplicated region for block: B:128:0x01ba  */
        /* JADX WARN: Removed duplicated region for block: B:130:0x01bd  */
        /* JADX WARN: Removed duplicated region for block: B:131:0x01bf  */
        /* JADX WARN: Removed duplicated region for block: B:155:0x01ea A[RETURN] */
        /* JADX WARN: Removed duplicated region for block: B:156:0x01eb  */
        @SuppressLint({"WrongCall"})
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public final void b(b.h.b.i.d dVar, b.a aVar) {
            int makeMeasureSpec;
            int i;
            e eVar;
            int max;
            int i2;
            int i3;
            int i4;
            int i5;
            boolean z;
            int baseline;
            int i6;
            if (dVar == null) {
                return;
            }
            int i7 = 0;
            if (dVar.c0 == 8 && !dVar.z) {
                aVar.f1891e = 0;
                aVar.f1892f = 0;
                aVar.f1893g = 0;
            } else if (dVar.P == null) {
            } else {
                int i8 = aVar.f1887a;
                int i9 = aVar.f1888b;
                int i10 = aVar.f1889c;
                int i11 = aVar.f1890d;
                int i12 = this.f207b + this.f208c;
                int i13 = this.f209d;
                View view = (View) dVar.b0;
                int f2 = m0.f(i8);
                if (f2 == 0) {
                    i7 = View.MeasureSpec.makeMeasureSpec(i10, 1073741824);
                } else if (f2 == 1) {
                    i7 = ViewGroup.getChildMeasureSpec(this.f211f, i13, -2);
                } else if (f2 == 2) {
                    i7 = ViewGroup.getChildMeasureSpec(this.f211f, i13, -2);
                    boolean z2 = dVar.l == 1;
                    int i14 = aVar.j;
                    if (i14 == 1 || i14 == 2) {
                        if (aVar.j == 2 || !z2 || (z2 && (view.getMeasuredHeight() == dVar.l())) || (view instanceof g) || dVar.z()) {
                            i7 = View.MeasureSpec.makeMeasureSpec(dVar.r(), 1073741824);
                        }
                    }
                } else if (f2 == 3) {
                    int i15 = this.f211f;
                    b.h.b.i.c cVar = dVar.D;
                    int i16 = cVar != null ? cVar.f1866g + 0 : 0;
                    b.h.b.i.c cVar2 = dVar.F;
                    if (cVar2 != null) {
                        i16 += cVar2.f1866g;
                    }
                    i7 = ViewGroup.getChildMeasureSpec(i15, i13 + i16, -1);
                }
                int f3 = m0.f(i9);
                if (f3 == 0) {
                    makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i11, 1073741824);
                } else if (f3 == 1) {
                    makeMeasureSpec = ViewGroup.getChildMeasureSpec(this.f212g, i12, -2);
                } else if (f3 == 2) {
                    makeMeasureSpec = ViewGroup.getChildMeasureSpec(this.f212g, i12, -2);
                    boolean z3 = dVar.m == 1;
                    int i17 = aVar.j;
                    if (i17 == 1 || i17 == 2) {
                        if (aVar.j == 2 || !z3 || (z3 && (view.getMeasuredWidth() == dVar.r())) || (view instanceof g) || dVar.A()) {
                            makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(dVar.l(), 1073741824);
                        }
                    }
                } else if (f3 == 3) {
                    int i18 = this.f212g;
                    int i19 = dVar.D != null ? dVar.E.f1866g + 0 : 0;
                    if (dVar.F != null) {
                        i19 += dVar.G.f1866g;
                    }
                    makeMeasureSpec = ViewGroup.getChildMeasureSpec(i18, i12 + i19, -1);
                } else {
                    i = 0;
                    eVar = (e) dVar.P;
                    if (eVar != null && i.b(ConstraintLayout.this.mOptimizationLevel, 256) && view.getMeasuredWidth() == dVar.r() && view.getMeasuredWidth() < eVar.r() && view.getMeasuredHeight() == dVar.l() && view.getMeasuredHeight() < eVar.l() && view.getBaseline() == dVar.W && !dVar.y()) {
                        if (!a(dVar.B, i7, dVar.r()) && a(dVar.C, i, dVar.l())) {
                            aVar.f1891e = dVar.r();
                            aVar.f1892f = dVar.l();
                            aVar.f1893g = dVar.W;
                            return;
                        }
                    }
                    boolean z4 = i8 != 3;
                    boolean z5 = i9 != 3;
                    boolean z6 = i9 != 4 || i9 == 1;
                    boolean z7 = i8 != 4 || i8 == 1;
                    boolean z8 = !z4 && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    boolean z9 = !z5 && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    if (view != null) {
                        return;
                    }
                    a aVar2 = (a) view.getLayoutParams();
                    int i20 = aVar.j;
                    if (i20 != 1 && i20 != 2 && z4 && dVar.l == 0 && z5 && dVar.m == 0) {
                        i6 = -1;
                        baseline = 0;
                        z = false;
                        max = 0;
                        i3 = 0;
                    } else {
                        if ((view instanceof j) && (dVar instanceof b.h.b.i.j)) {
                            b.h.b.i.j jVar = (b.h.b.i.j) dVar;
                            j jVar2 = (j) view;
                        } else {
                            view.measure(i7, i);
                        }
                        dVar.B = i7;
                        dVar.C = i;
                        dVar.f1881g = false;
                        int measuredWidth = view.getMeasuredWidth();
                        int measuredHeight = view.getMeasuredHeight();
                        int baseline2 = view.getBaseline();
                        int i21 = dVar.o;
                        max = i21 > 0 ? Math.max(i21, measuredWidth) : measuredWidth;
                        int i22 = dVar.p;
                        if (i22 > 0) {
                            max = Math.min(i22, max);
                        }
                        int i23 = dVar.r;
                        if (i23 > 0) {
                            i3 = Math.max(i23, measuredHeight);
                            i2 = i7;
                        } else {
                            i2 = i7;
                            i3 = measuredHeight;
                        }
                        int i24 = dVar.s;
                        if (i24 > 0) {
                            i3 = Math.min(i24, i3);
                        }
                        if (!i.b(ConstraintLayout.this.mOptimizationLevel, 1)) {
                            if (z8 && z6) {
                                max = (int) ((i3 * dVar.S) + 0.5f);
                            } else if (z9 && z7) {
                                i3 = (int) ((max / dVar.S) + 0.5f);
                            }
                        }
                        if (measuredWidth == max && measuredHeight == i3) {
                            baseline = baseline2;
                            i6 = -1;
                            z = false;
                        } else {
                            if (measuredWidth != max) {
                                i4 = 1073741824;
                                i5 = View.MeasureSpec.makeMeasureSpec(max, 1073741824);
                            } else {
                                i4 = 1073741824;
                                i5 = i2;
                            }
                            if (measuredHeight != i3) {
                                i = View.MeasureSpec.makeMeasureSpec(i3, i4);
                            }
                            view.measure(i5, i);
                            dVar.B = i5;
                            dVar.C = i;
                            z = false;
                            dVar.f1881g = false;
                            int measuredWidth2 = view.getMeasuredWidth();
                            int measuredHeight2 = view.getMeasuredHeight();
                            baseline = view.getBaseline();
                            max = measuredWidth2;
                            i3 = measuredHeight2;
                            i6 = -1;
                        }
                    }
                    boolean z10 = baseline != i6 ? true : z;
                    aVar.i = (max == aVar.f1889c && i3 == aVar.f1890d) ? z : true;
                    if (aVar2.X) {
                        z10 = true;
                    }
                    if (z10 && baseline != -1 && dVar.W != baseline) {
                        aVar.i = true;
                    }
                    aVar.f1891e = max;
                    aVar.f1892f = i3;
                    aVar.f1894h = z10;
                    aVar.f1893g = baseline;
                    return;
                }
                i = makeMeasureSpec;
                eVar = (e) dVar.P;
                if (eVar != null) {
                    if (!a(dVar.B, i7, dVar.r()) && a(dVar.C, i, dVar.l())) {
                    }
                }
                if (i8 != 3) {
                }
                if (i9 != 3) {
                }
                if (i9 != 4) {
                }
                if (i8 != 4) {
                }
                if (z4) {
                }
                if (z5) {
                }
                if (view != null) {
                }
            }
        }
    }

    public ConstraintLayout(Context context) {
        super(context);
        this.mChildrenByIds = new SparseArray<>();
        this.mConstraintHelpers = new ArrayList<>(4);
        this.mLayoutWidget = new e();
        this.mMinWidth = 0;
        this.mMinHeight = 0;
        this.mMaxWidth = Integer.MAX_VALUE;
        this.mMaxHeight = Integer.MAX_VALUE;
        this.mDirtyHierarchy = true;
        this.mOptimizationLevel = Imgcodecs.IMWRITE_TIFF_XDPI;
        this.mConstraintSet = null;
        this.mConstraintLayoutSpec = null;
        this.mConstraintSetId = -1;
        this.mDesignIds = new HashMap<>();
        this.mLastMeasureWidth = -1;
        this.mLastMeasureHeight = -1;
        this.mLastMeasureWidthSize = -1;
        this.mLastMeasureHeightSize = -1;
        this.mLastMeasureWidthMode = 0;
        this.mLastMeasureHeightMode = 0;
        this.mTempMapIdToWidget = new SparseArray<>();
        this.mMeasurer = new b(this);
        this.mOnMeasureWidthMeasureSpec = 0;
        this.mOnMeasureHeightMeasureSpec = 0;
        init(null, 0, 0);
    }

    private int getPaddingWidth() {
        int max = Math.max(0, getPaddingRight()) + Math.max(0, getPaddingLeft());
        int max2 = Math.max(0, getPaddingEnd()) + Math.max(0, getPaddingStart());
        return max2 > 0 ? max2 : max;
    }

    private final b.h.b.i.d getTargetWidget(int i) {
        if (i == 0) {
            return this.mLayoutWidget;
        }
        View view = this.mChildrenByIds.get(i);
        if (view == null && (view = findViewById(i)) != null && view != this && view.getParent() == this) {
            onViewAdded(view);
        }
        if (view == this) {
            return this.mLayoutWidget;
        }
        if (view == null) {
            return null;
        }
        return ((a) view.getLayoutParams()).l0;
    }

    private void init(AttributeSet attributeSet, int i, int i2) {
        e eVar = this.mLayoutWidget;
        eVar.b0 = this;
        b bVar = this.mMeasurer;
        eVar.o0 = bVar;
        eVar.n0.f1900f = bVar;
        this.mChildrenByIds.put(getId(), this);
        this.mConstraintSet = null;
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, b.h.c.i.f2010b, i, i2);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i3 = 0; i3 < indexCount; i3++) {
                int index = obtainStyledAttributes.getIndex(i3);
                if (index == 9) {
                    this.mMinWidth = obtainStyledAttributes.getDimensionPixelOffset(index, this.mMinWidth);
                } else if (index == 10) {
                    this.mMinHeight = obtainStyledAttributes.getDimensionPixelOffset(index, this.mMinHeight);
                } else if (index == 7) {
                    this.mMaxWidth = obtainStyledAttributes.getDimensionPixelOffset(index, this.mMaxWidth);
                } else if (index == 8) {
                    this.mMaxHeight = obtainStyledAttributes.getDimensionPixelOffset(index, this.mMaxHeight);
                } else if (index == 90) {
                    this.mOptimizationLevel = obtainStyledAttributes.getInt(index, this.mOptimizationLevel);
                } else if (index == 39) {
                    int resourceId = obtainStyledAttributes.getResourceId(index, 0);
                    if (resourceId != 0) {
                        try {
                            parseLayoutDescription(resourceId);
                        } catch (Resources.NotFoundException unused) {
                            this.mConstraintLayoutSpec = null;
                        }
                    }
                } else if (index == 18) {
                    int resourceId2 = obtainStyledAttributes.getResourceId(index, 0);
                    try {
                        d dVar = new d();
                        this.mConstraintSet = dVar;
                        dVar.f(getContext(), resourceId2);
                    } catch (Resources.NotFoundException unused2) {
                        this.mConstraintSet = null;
                    }
                    this.mConstraintSetId = resourceId2;
                }
            }
            obtainStyledAttributes.recycle();
        }
        this.mLayoutWidget.Z(this.mOptimizationLevel);
    }

    private void markHierarchyDirty() {
        this.mDirtyHierarchy = true;
        this.mLastMeasureWidth = -1;
        this.mLastMeasureHeight = -1;
        this.mLastMeasureWidthSize = -1;
        this.mLastMeasureHeightSize = -1;
        this.mLastMeasureWidthMode = 0;
        this.mLastMeasureHeightMode = 0;
    }

    private void setChildrenConstraints() {
        String str;
        int e2;
        boolean isInEditMode = isInEditMode();
        int childCount = getChildCount();
        for (int i = 0; i < childCount; i++) {
            b.h.b.i.d viewWidget = getViewWidget(getChildAt(i));
            if (viewWidget != null) {
                viewWidget.B();
            }
        }
        if (isInEditMode) {
            for (int i2 = 0; i2 < childCount; i2++) {
                View childAt = getChildAt(i2);
                try {
                    String resourceName = getResources().getResourceName(childAt.getId());
                    setDesignInformation(0, resourceName, Integer.valueOf(childAt.getId()));
                    int indexOf = resourceName.indexOf(47);
                    if (indexOf != -1) {
                        resourceName = resourceName.substring(indexOf + 1);
                    }
                    getTargetWidget(childAt.getId()).d0 = resourceName;
                } catch (Resources.NotFoundException unused) {
                }
            }
        }
        if (this.mConstraintSetId != -1) {
            for (int i3 = 0; i3 < childCount; i3++) {
                View childAt2 = getChildAt(i3);
                if (childAt2.getId() == this.mConstraintSetId && (childAt2 instanceof b.h.c.e)) {
                    this.mConstraintSet = ((b.h.c.e) childAt2).getConstraintSet();
                }
            }
        }
        d dVar = this.mConstraintSet;
        if (dVar != null) {
            dVar.b(this, true);
        }
        this.mLayoutWidget.l0.clear();
        int size = this.mConstraintHelpers.size();
        if (size > 0) {
            for (int i4 = 0; i4 < size; i4++) {
                b.h.c.b bVar = this.mConstraintHelpers.get(i4);
                if (bVar.isInEditMode()) {
                    bVar.setIds(bVar.f1947f);
                }
                b.h.b.i.g gVar = bVar.f1946e;
                if (gVar != null) {
                    h hVar = (h) gVar;
                    hVar.m0 = 0;
                    Arrays.fill(hVar.l0, (Object) null);
                    for (int i5 = 0; i5 < bVar.f1944c; i5++) {
                        int i6 = bVar.f1943b[i5];
                        View viewById = getViewById(i6);
                        if (viewById == null && (e2 = bVar.e(this, (str = bVar.f1949h.get(Integer.valueOf(i6))))) != 0) {
                            bVar.f1943b[i5] = e2;
                            bVar.f1949h.put(Integer.valueOf(e2), str);
                            viewById = getViewById(e2);
                        }
                        if (viewById != null) {
                            b.h.b.i.g gVar2 = bVar.f1946e;
                            b.h.b.i.d viewWidget2 = getViewWidget(viewById);
                            h hVar2 = (h) gVar2;
                            Objects.requireNonNull(hVar2);
                            if (viewWidget2 != hVar2 && viewWidget2 != null) {
                                int i7 = hVar2.m0 + 1;
                                b.h.b.i.d[] dVarArr = hVar2.l0;
                                if (i7 > dVarArr.length) {
                                    hVar2.l0 = (b.h.b.i.d[]) Arrays.copyOf(dVarArr, dVarArr.length * 2);
                                }
                                b.h.b.i.d[] dVarArr2 = hVar2.l0;
                                int i8 = hVar2.m0;
                                dVarArr2[i8] = viewWidget2;
                                hVar2.m0 = i8 + 1;
                            }
                        }
                    }
                    bVar.f1946e.a(this.mLayoutWidget);
                }
            }
        }
        for (int i9 = 0; i9 < childCount; i9++) {
            View childAt3 = getChildAt(i9);
            if (childAt3 instanceof g) {
                g gVar3 = (g) childAt3;
                if (gVar3.f2006b == -1 && !gVar3.isInEditMode()) {
                    gVar3.setVisibility(gVar3.f2008d);
                }
                View findViewById = findViewById(gVar3.f2006b);
                gVar3.f2007c = findViewById;
                if (findViewById != null) {
                    ((a) findViewById.getLayoutParams()).a0 = true;
                    gVar3.f2007c.setVisibility(0);
                    gVar3.setVisibility(0);
                }
            }
        }
        this.mTempMapIdToWidget.clear();
        this.mTempMapIdToWidget.put(0, this.mLayoutWidget);
        this.mTempMapIdToWidget.put(getId(), this.mLayoutWidget);
        for (int i10 = 0; i10 < childCount; i10++) {
            View childAt4 = getChildAt(i10);
            this.mTempMapIdToWidget.put(childAt4.getId(), getViewWidget(childAt4));
        }
        for (int i11 = 0; i11 < childCount; i11++) {
            View childAt5 = getChildAt(i11);
            b.h.b.i.d viewWidget3 = getViewWidget(childAt5);
            if (viewWidget3 != null) {
                a aVar = (a) childAt5.getLayoutParams();
                e eVar = this.mLayoutWidget;
                eVar.l0.add(viewWidget3);
                b.h.b.i.d dVar2 = viewWidget3.P;
                if (dVar2 != null) {
                    ((k) dVar2).l0.remove(viewWidget3);
                    viewWidget3.B();
                }
                viewWidget3.P = eVar;
                applyConstraintsFromLayoutParams(isInEditMode, childAt5, viewWidget3, aVar, this.mTempMapIdToWidget);
            }
        }
    }

    private boolean updateHierarchy() {
        int childCount = getChildCount();
        boolean z = false;
        int i = 0;
        while (true) {
            if (i >= childCount) {
                break;
            } else if (getChildAt(i).isLayoutRequested()) {
                z = true;
                break;
            } else {
                i++;
            }
        }
        if (z) {
            setChildrenConstraints();
        }
        return z;
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i, ViewGroup.LayoutParams layoutParams) {
        super.addView(view, i, layoutParams);
    }

    /* JADX WARN: Removed duplicated region for block: B:37:0x00d4  */
    /* JADX WARN: Removed duplicated region for block: B:40:0x00eb  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x0108  */
    /* JADX WARN: Removed duplicated region for block: B:50:0x0121  */
    /* JADX WARN: Removed duplicated region for block: B:57:0x0143  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x015c  */
    /* JADX WARN: Removed duplicated region for block: B:67:0x017e  */
    /* JADX WARN: Removed duplicated region for block: B:75:0x01cd  */
    /* JADX WARN: Removed duplicated region for block: B:78:0x01d7  */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:150:0x030b -> B:151:0x030c). Please submit an issue!!! */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void applyConstraintsFromLayoutParams(boolean z, View view, b.h.b.i.d dVar, a aVar, SparseArray<b.h.b.i.d> sparseArray) {
        int i;
        b.h.b.i.d dVar2;
        int i2;
        b.h.b.i.d dVar3;
        int i3;
        b.h.b.i.d dVar4;
        int i4;
        float f2;
        int i5;
        int i6;
        int i7;
        float f3;
        int i8;
        int i9;
        c.a aVar2 = c.a.RIGHT;
        c.a aVar3 = c.a.LEFT;
        c.a aVar4 = c.a.BOTTOM;
        c.a aVar5 = c.a.TOP;
        aVar.a();
        dVar.c0 = view.getVisibility();
        if (aVar.a0) {
            dVar.z = true;
            dVar.c0 = 8;
        }
        dVar.b0 = view;
        if (view instanceof b.h.c.b) {
            ((b.h.c.b) view).g(dVar, this.mLayoutWidget.p0);
        }
        if (aVar.Y) {
            b.h.b.i.f fVar = (b.h.b.i.f) dVar;
            int i10 = aVar.i0;
            int i11 = aVar.j0;
            float f4 = aVar.k0;
            int i12 = (f4 > (-1.0f) ? 1 : (f4 == (-1.0f) ? 0 : -1));
            if (i12 != 0) {
                if (i12 > 0) {
                    fVar.l0 = f4;
                    fVar.m0 = -1;
                    fVar.n0 = -1;
                    return;
                }
                return;
            } else if (i10 != -1) {
                if (i10 > -1) {
                    fVar.l0 = -1.0f;
                    fVar.m0 = i10;
                    fVar.n0 = -1;
                    return;
                }
                return;
            } else if (i11 == -1 || i11 <= -1) {
                return;
            } else {
                fVar.l0 = -1.0f;
                fVar.m0 = -1;
                fVar.n0 = i11;
                return;
            }
        }
        int i13 = aVar.b0;
        int i14 = aVar.c0;
        int i15 = aVar.d0;
        int i16 = aVar.e0;
        int i17 = aVar.f0;
        int i18 = aVar.g0;
        float f5 = aVar.h0;
        int i19 = aVar.m;
        if (i19 != -1) {
            b.h.b.i.d dVar5 = sparseArray.get(i19);
            if (dVar5 != null) {
                float f6 = aVar.o;
                int i20 = aVar.n;
                c.a aVar6 = c.a.CENTER;
                dVar.i(aVar6).a(dVar5.i(aVar6), i20, 0, true);
                dVar.x = f6;
            }
        } else {
            if (i13 != -1) {
                b.h.b.i.d dVar6 = sparseArray.get(i13);
                if (dVar6 != null) {
                    dVar.i(aVar3).a(dVar6.i(aVar3), ((ViewGroup.MarginLayoutParams) aVar).leftMargin, i17, true);
                }
            } else {
                i = -1;
                if (i14 != -1) {
                    b.h.b.i.d dVar7 = sparseArray.get(i14);
                    if (dVar7 != null) {
                        dVar.i(aVar3).a(dVar7.i(aVar2), ((ViewGroup.MarginLayoutParams) aVar).leftMargin, i17, true);
                    }
                }
                if (i15 == i) {
                    b.h.b.i.d dVar8 = sparseArray.get(i15);
                    if (dVar8 != null) {
                        dVar.i(aVar2).a(dVar8.i(aVar3), ((ViewGroup.MarginLayoutParams) aVar).rightMargin, i18, true);
                    }
                } else if (i16 != i && (dVar2 = sparseArray.get(i16)) != null) {
                    dVar.i(aVar2).a(dVar2.i(aVar2), ((ViewGroup.MarginLayoutParams) aVar).rightMargin, i18, true);
                }
                i2 = aVar.f204h;
                if (i2 == -1) {
                    b.h.b.i.d dVar9 = sparseArray.get(i2);
                    if (dVar9 != null) {
                        dVar.i(aVar5).a(dVar9.i(aVar5), ((ViewGroup.MarginLayoutParams) aVar).topMargin, aVar.u, true);
                    }
                } else {
                    int i21 = aVar.i;
                    if (i21 != -1 && (dVar3 = sparseArray.get(i21)) != null) {
                        dVar.i(aVar5).a(dVar3.i(aVar4), ((ViewGroup.MarginLayoutParams) aVar).topMargin, aVar.u, true);
                    }
                }
                i3 = aVar.j;
                if (i3 == -1) {
                    b.h.b.i.d dVar10 = sparseArray.get(i3);
                    if (dVar10 != null) {
                        dVar.i(aVar4).a(dVar10.i(aVar5), ((ViewGroup.MarginLayoutParams) aVar).bottomMargin, aVar.w, true);
                    }
                } else {
                    int i22 = aVar.k;
                    if (i22 != -1 && (dVar4 = sparseArray.get(i22)) != null) {
                        dVar.i(aVar4).a(dVar4.i(aVar4), ((ViewGroup.MarginLayoutParams) aVar).bottomMargin, aVar.w, true);
                    }
                }
                i4 = aVar.l;
                if (i4 != -1) {
                    View view2 = this.mChildrenByIds.get(i4);
                    b.h.b.i.d dVar11 = sparseArray.get(aVar.l);
                    if (dVar11 != null && view2 != null && (view2.getLayoutParams() instanceof a)) {
                        a aVar7 = (a) view2.getLayoutParams();
                        aVar.X = true;
                        aVar7.X = true;
                        c.a aVar8 = c.a.BASELINE;
                        dVar.i(aVar8).a(dVar11.i(aVar8), 0, -1, true);
                        dVar.y = true;
                        aVar7.l0.y = true;
                        dVar.i(aVar5).h();
                        dVar.i(aVar4).h();
                    }
                }
                if (f5 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    dVar.Z = f5;
                }
                f2 = aVar.A;
                if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    dVar.a0 = f2;
                }
            }
            i = -1;
            if (i15 == i) {
            }
            i2 = aVar.f204h;
            if (i2 == -1) {
            }
            i3 = aVar.j;
            if (i3 == -1) {
            }
            i4 = aVar.l;
            if (i4 != -1) {
            }
            if (f5 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            }
            f2 = aVar.A;
            if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            }
        }
        if (z && ((i9 = aVar.P) != -1 || aVar.Q != -1)) {
            int i23 = aVar.Q;
            dVar.U = i9;
            dVar.V = i23;
        }
        if (!aVar.V) {
            if (((ViewGroup.MarginLayoutParams) aVar).width == -1) {
                if (aVar.S) {
                    dVar.O[0] = 3;
                } else {
                    dVar.O[0] = 4;
                }
                dVar.i(aVar3).f1866g = ((ViewGroup.MarginLayoutParams) aVar).leftMargin;
                dVar.i(aVar2).f1866g = ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
            } else {
                dVar.O[0] = 3;
                dVar.M(0);
            }
        } else {
            dVar.O[0] = 1;
            dVar.M(((ViewGroup.MarginLayoutParams) aVar).width);
            if (((ViewGroup.MarginLayoutParams) aVar).width == -2) {
                dVar.O[0] = 2;
            }
        }
        if (!aVar.W) {
            i5 = -1;
            if (((ViewGroup.MarginLayoutParams) aVar).height == -1) {
                if (aVar.T) {
                    dVar.O[1] = 3;
                } else {
                    dVar.O[1] = 4;
                }
                dVar.i(aVar5).f1866g = ((ViewGroup.MarginLayoutParams) aVar).topMargin;
                dVar.i(aVar4).f1866g = ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
            } else {
                dVar.O[1] = 3;
                dVar.H(0);
            }
        } else {
            i5 = -1;
            dVar.O[1] = 1;
            dVar.H(((ViewGroup.MarginLayoutParams) aVar).height);
            if (((ViewGroup.MarginLayoutParams) aVar).height == -2) {
                dVar.O[1] = 2;
            }
        }
        String str = aVar.B;
        if (str != null && str.length() != 0) {
            int length = str.length();
            int indexOf = str.indexOf(44);
            if (indexOf <= 0 || indexOf >= length - 1) {
                i6 = 1;
                i7 = 0;
            } else {
                String substring = str.substring(0, indexOf);
                if (substring.equalsIgnoreCase("W")) {
                    i6 = 1;
                    i8 = 0;
                } else if (substring.equalsIgnoreCase("H")) {
                    i6 = 1;
                    i8 = 1;
                } else {
                    i8 = i5;
                    i6 = 1;
                }
                i7 = indexOf + 1;
                i5 = i8;
            }
            int indexOf2 = str.indexOf(58);
            if (indexOf2 >= 0 && indexOf2 < length - i6) {
                String substring2 = str.substring(i7, indexOf2);
                String substring3 = str.substring(indexOf2 + i6);
                if (substring2.length() > 0 && substring3.length() > 0) {
                    float parseFloat = Float.parseFloat(substring2);
                    float parseFloat2 = Float.parseFloat(substring3);
                    if (parseFloat > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && parseFloat2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        if (i5 == 1) {
                            f3 = Math.abs(parseFloat2 / parseFloat);
                        } else {
                            f3 = Math.abs(parseFloat / parseFloat2);
                        }
                    }
                }
                f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            } else {
                String substring4 = str.substring(i7);
                if (substring4.length() > 0) {
                    f3 = Float.parseFloat(substring4);
                }
                f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
            if (f3 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                dVar.S = f3;
                dVar.T = i5;
            }
        } else {
            dVar.S = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        float f7 = aVar.D;
        float[] fArr = dVar.g0;
        fArr[0] = f7;
        fArr[1] = aVar.E;
        dVar.e0 = aVar.F;
        dVar.f0 = aVar.G;
        int i24 = aVar.H;
        int i25 = aVar.J;
        int i26 = aVar.L;
        float f8 = aVar.N;
        dVar.l = i24;
        dVar.o = i25;
        if (i26 == Integer.MAX_VALUE) {
            i26 = 0;
        }
        dVar.p = i26;
        dVar.q = f8;
        if (f8 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f8 < 1.0f && i24 == 0) {
            dVar.l = 2;
        }
        int i27 = aVar.I;
        int i28 = aVar.K;
        int i29 = aVar.M;
        float f9 = aVar.O;
        dVar.m = i27;
        dVar.r = i28;
        dVar.s = i29 != Integer.MAX_VALUE ? i29 : 0;
        dVar.t = f9;
        if (f9 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || f9 >= 1.0f || i27 != 0) {
            return;
        }
        dVar.m = 2;
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return layoutParams instanceof a;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchDraw(Canvas canvas) {
        Object tag;
        int size;
        ArrayList<b.h.c.b> arrayList = this.mConstraintHelpers;
        if (arrayList != null && (size = arrayList.size()) > 0) {
            for (int i = 0; i < size; i++) {
                this.mConstraintHelpers.get(i).j();
            }
        }
        super.dispatchDraw(canvas);
        if (isInEditMode()) {
            int childCount = getChildCount();
            float width = getWidth();
            float height = getHeight();
            for (int i2 = 0; i2 < childCount; i2++) {
                View childAt = getChildAt(i2);
                if (childAt.getVisibility() != 8 && (tag = childAt.getTag()) != null && (tag instanceof String)) {
                    String[] split = ((String) tag).split(",");
                    if (split.length == 4) {
                        int parseInt = Integer.parseInt(split[0]);
                        int parseInt2 = Integer.parseInt(split[1]);
                        int parseInt3 = Integer.parseInt(split[2]);
                        int i3 = (int) ((parseInt / 1080.0f) * width);
                        int i4 = (int) ((parseInt2 / 1920.0f) * height);
                        Paint paint = new Paint();
                        paint.setColor(-65536);
                        float f2 = i3;
                        float f3 = i4;
                        float f4 = i3 + ((int) ((parseInt3 / 1080.0f) * width));
                        canvas.drawLine(f2, f3, f4, f3, paint);
                        float parseInt4 = i4 + ((int) ((Integer.parseInt(split[3]) / 1920.0f) * height));
                        canvas.drawLine(f4, f3, f4, parseInt4, paint);
                        canvas.drawLine(f4, parseInt4, f2, parseInt4, paint);
                        canvas.drawLine(f2, parseInt4, f2, f3, paint);
                        paint.setColor(-16711936);
                        canvas.drawLine(f2, f3, f4, parseInt4, paint);
                        canvas.drawLine(f2, parseInt4, f4, f3, paint);
                    }
                }
            }
        }
    }

    public void fillMetrics(b.h.b.e eVar) {
        Objects.requireNonNull(this.mLayoutWidget.q0);
    }

    @Override // android.view.View
    public void forceLayout() {
        markHierarchyDirty();
        super.forceLayout();
    }

    public Object getDesignInformation(int i, Object obj) {
        if (i == 0 && (obj instanceof String)) {
            String str = (String) obj;
            HashMap<String, Integer> hashMap = this.mDesignIds;
            if (hashMap == null || !hashMap.containsKey(str)) {
                return null;
            }
            return this.mDesignIds.get(str);
        }
        return null;
    }

    public int getMaxHeight() {
        return this.mMaxHeight;
    }

    public int getMaxWidth() {
        return this.mMaxWidth;
    }

    public int getMinHeight() {
        return this.mMinHeight;
    }

    public int getMinWidth() {
        return this.mMinWidth;
    }

    public int getOptimizationLevel() {
        return this.mLayoutWidget.x0;
    }

    public View getViewById(int i) {
        return this.mChildrenByIds.get(i);
    }

    public final b.h.b.i.d getViewWidget(View view) {
        if (view == this) {
            return this.mLayoutWidget;
        }
        if (view == null) {
            return null;
        }
        return ((a) view.getLayoutParams()).l0;
    }

    public boolean isRtl() {
        return ((getContext().getApplicationInfo().flags & Calib3d.CALIB_USE_EXTRINSIC_GUESS) != 0) && 1 == getLayoutDirection();
    }

    public void loadLayoutDescription(int i) {
        if (i != 0) {
            try {
                this.mConstraintLayoutSpec = new b.h.c.c(getContext(), this, i);
                return;
            } catch (Resources.NotFoundException unused) {
                this.mConstraintLayoutSpec = null;
                return;
            }
        }
        this.mConstraintLayoutSpec = null;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        View content;
        int childCount = getChildCount();
        boolean isInEditMode = isInEditMode();
        for (int i5 = 0; i5 < childCount; i5++) {
            View childAt = getChildAt(i5);
            a aVar = (a) childAt.getLayoutParams();
            b.h.b.i.d dVar = aVar.l0;
            if ((childAt.getVisibility() != 8 || aVar.Y || aVar.Z || isInEditMode) && !aVar.a0) {
                int s = dVar.s();
                int t = dVar.t();
                int r = dVar.r() + s;
                int l = dVar.l() + t;
                childAt.layout(s, t, r, l);
                if ((childAt instanceof g) && (content = ((g) childAt).getContent()) != null) {
                    content.setVisibility(0);
                    content.layout(s, t, r, l);
                }
            }
        }
        int size = this.mConstraintHelpers.size();
        if (size > 0) {
            for (int i6 = 0; i6 < size; i6++) {
                this.mConstraintHelpers.get(i6).h();
            }
        }
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        if (!this.mDirtyHierarchy) {
            int childCount = getChildCount();
            int i3 = 0;
            while (true) {
                if (i3 >= childCount) {
                    break;
                } else if (getChildAt(i3).isLayoutRequested()) {
                    this.mDirtyHierarchy = true;
                    break;
                } else {
                    i3++;
                }
            }
        }
        if (!this.mDirtyHierarchy) {
            int i4 = this.mOnMeasureWidthMeasureSpec;
            if (i4 == i && this.mOnMeasureHeightMeasureSpec == i2) {
                int r = this.mLayoutWidget.r();
                int l = this.mLayoutWidget.l();
                e eVar = this.mLayoutWidget;
                resolveMeasuredDimension(i, i2, r, l, eVar.y0, eVar.z0);
                return;
            } else if (i4 == i && View.MeasureSpec.getMode(i) == 1073741824 && View.MeasureSpec.getMode(i2) == Integer.MIN_VALUE && View.MeasureSpec.getMode(this.mOnMeasureHeightMeasureSpec) == Integer.MIN_VALUE && View.MeasureSpec.getSize(i2) >= this.mLayoutWidget.l()) {
                this.mOnMeasureWidthMeasureSpec = i;
                this.mOnMeasureHeightMeasureSpec = i2;
                int r2 = this.mLayoutWidget.r();
                int l2 = this.mLayoutWidget.l();
                e eVar2 = this.mLayoutWidget;
                resolveMeasuredDimension(i, i2, r2, l2, eVar2.y0, eVar2.z0);
                return;
            }
        }
        this.mOnMeasureWidthMeasureSpec = i;
        this.mOnMeasureHeightMeasureSpec = i2;
        this.mLayoutWidget.p0 = isRtl();
        if (this.mDirtyHierarchy) {
            this.mDirtyHierarchy = false;
            if (updateHierarchy()) {
                e eVar3 = this.mLayoutWidget;
                eVar3.m0.c(eVar3);
            }
        }
        resolveSystem(this.mLayoutWidget, this.mOptimizationLevel, i, i2);
        int r3 = this.mLayoutWidget.r();
        int l3 = this.mLayoutWidget.l();
        e eVar4 = this.mLayoutWidget;
        resolveMeasuredDimension(i, i2, r3, l3, eVar4.y0, eVar4.z0);
    }

    @Override // android.view.ViewGroup
    public void onViewAdded(View view) {
        super.onViewAdded(view);
        b.h.b.i.d viewWidget = getViewWidget(view);
        if ((view instanceof Guideline) && !(viewWidget instanceof b.h.b.i.f)) {
            a aVar = (a) view.getLayoutParams();
            b.h.b.i.f fVar = new b.h.b.i.f();
            aVar.l0 = fVar;
            aVar.Y = true;
            fVar.Q(aVar.R);
        }
        if (view instanceof b.h.c.b) {
            b.h.c.b bVar = (b.h.c.b) view;
            bVar.k();
            ((a) view.getLayoutParams()).Z = true;
            if (!this.mConstraintHelpers.contains(bVar)) {
                this.mConstraintHelpers.add(bVar);
            }
        }
        this.mChildrenByIds.put(view.getId(), view);
        this.mDirtyHierarchy = true;
    }

    @Override // android.view.ViewGroup
    public void onViewRemoved(View view) {
        super.onViewRemoved(view);
        this.mChildrenByIds.remove(view.getId());
        b.h.b.i.d viewWidget = getViewWidget(view);
        this.mLayoutWidget.l0.remove(viewWidget);
        viewWidget.B();
        this.mConstraintHelpers.remove(view);
        this.mDirtyHierarchy = true;
    }

    public void parseLayoutDescription(int i) {
        this.mConstraintLayoutSpec = new b.h.c.c(getContext(), this, i);
    }

    @Override // android.view.ViewGroup, android.view.ViewManager
    public void removeView(View view) {
        super.removeView(view);
    }

    @Override // android.view.View, android.view.ViewParent
    public void requestLayout() {
        markHierarchyDirty();
        super.requestLayout();
    }

    public void resolveMeasuredDimension(int i, int i2, int i3, int i4, boolean z, boolean z2) {
        b bVar = this.mMeasurer;
        int i5 = bVar.f210e;
        int resolveSizeAndState = ViewGroup.resolveSizeAndState(i3 + bVar.f209d, i, 0);
        int min = Math.min(this.mMaxWidth, resolveSizeAndState & 16777215);
        int min2 = Math.min(this.mMaxHeight, ViewGroup.resolveSizeAndState(i4 + i5, i2, 0) & 16777215);
        if (z) {
            min |= 16777216;
        }
        if (z2) {
            min2 |= 16777216;
        }
        setMeasuredDimension(min, min2);
        this.mLastMeasureWidth = min;
        this.mLastMeasureHeight = min2;
    }

    /* JADX WARN: Removed duplicated region for block: B:178:0x03c4  */
    /* JADX WARN: Removed duplicated region for block: B:184:0x03d7  */
    /* JADX WARN: Removed duplicated region for block: B:20:0x00aa  */
    /* JADX WARN: Removed duplicated region for block: B:266:0x04ff  */
    /* JADX WARN: Removed duplicated region for block: B:268:0x0504  */
    /* JADX WARN: Removed duplicated region for block: B:366:0x06e0  */
    /* JADX WARN: Removed duplicated region for block: B:63:0x0120  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void resolveSystem(e eVar, int i, int i2, int i3) {
        int i4;
        b.h.b.i.l.b bVar;
        int size;
        boolean z;
        c.a aVar;
        c.a aVar2;
        boolean z2;
        boolean z3;
        boolean z4;
        b.InterfaceC0029b interfaceC0029b;
        int i5;
        int i6;
        boolean z5;
        int size2;
        int i7;
        e eVar2;
        int i8;
        boolean z6;
        boolean z7;
        boolean z8;
        int i9;
        int i10;
        b.InterfaceC0029b interfaceC0029b2;
        int i11;
        b.InterfaceC0029b interfaceC0029b3;
        c.a aVar3;
        boolean z9;
        c.a aVar4;
        b.h.b.i.l.k kVar;
        m mVar;
        int i12;
        int i13;
        int i14;
        boolean z10;
        int i15;
        boolean z11;
        boolean z12;
        e eVar3 = eVar;
        int mode = View.MeasureSpec.getMode(i2);
        int size3 = View.MeasureSpec.getSize(i2);
        int mode2 = View.MeasureSpec.getMode(i3);
        int size4 = View.MeasureSpec.getSize(i3);
        int max = Math.max(0, getPaddingTop());
        int max2 = Math.max(0, getPaddingBottom());
        int i16 = max + max2;
        int paddingWidth = getPaddingWidth();
        b bVar2 = this.mMeasurer;
        bVar2.f207b = max;
        bVar2.f208c = max2;
        bVar2.f209d = paddingWidth;
        bVar2.f210e = i16;
        bVar2.f211f = i2;
        bVar2.f212g = i3;
        int max3 = Math.max(0, getPaddingStart());
        int max4 = Math.max(0, getPaddingEnd());
        if (max3 <= 0 && max4 <= 0) {
            max3 = Math.max(0, getPaddingLeft());
        } else if (isRtl()) {
            i4 = max4;
            int i17 = size3 - paddingWidth;
            int i18 = size4 - i16;
            setSelfDimensionBehaviour(eVar, mode, i17, mode2, i18);
            eVar3.r0 = i4;
            eVar3.s0 = max;
            bVar = eVar3.m0;
            Objects.requireNonNull(bVar);
            c.a aVar5 = c.a.BOTTOM;
            c.a aVar6 = c.a.RIGHT;
            b.InterfaceC0029b interfaceC0029b4 = eVar3.o0;
            size = eVar3.l0.size();
            int r = eVar.r();
            int l = eVar.l();
            boolean b2 = i.b(i, 128);
            z = !b2 || i.b(i, 64);
            if (z) {
                int i19 = 0;
                while (i19 < size) {
                    b.h.b.i.d dVar = eVar3.l0.get(i19);
                    boolean z13 = z;
                    aVar = aVar5;
                    aVar2 = aVar6;
                    boolean z14 = (dVar.m() == 3) && (dVar.q() == 3) && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    if ((dVar.w() && z14) || ((dVar.x() && z14) || (dVar instanceof b.h.b.i.j) || dVar.w() || dVar.x())) {
                        z2 = false;
                        break;
                    }
                    i19++;
                    z = z13;
                    aVar5 = aVar;
                    aVar6 = aVar2;
                }
            }
            aVar = aVar5;
            aVar2 = aVar6;
            z2 = z;
            z3 = z2 & ((mode != 1073741824 && mode2 == 1073741824) || b2);
            if (z3) {
                z4 = z3;
                interfaceC0029b = interfaceC0029b4;
                i5 = r;
                i6 = 0;
                z5 = false;
            } else {
                int min = Math.min(eVar3.w[0], i17);
                int min2 = Math.min(eVar3.w[1], i18);
                if (mode == 1073741824 && eVar.r() != min) {
                    eVar3.M(min);
                    eVar.W();
                }
                if (mode2 == 1073741824 && eVar.l() != min2) {
                    eVar3.H(min2);
                    eVar.W();
                }
                if (mode == 1073741824 && mode2 == 1073741824) {
                    b.h.b.i.l.e eVar4 = eVar3.n0;
                    boolean z15 = b2 & true;
                    if (eVar4.f1896b || eVar4.f1897c) {
                        Iterator<b.h.b.i.d> it = eVar4.f1895a.l0.iterator();
                        while (it.hasNext()) {
                            b.h.b.i.d next = it.next();
                            next.h();
                            next.f1875a = false;
                            next.f1878d.n();
                            next.f1879e.m();
                        }
                        i15 = 0;
                        eVar4.f1895a.h();
                        e eVar5 = eVar4.f1895a;
                        eVar5.f1875a = false;
                        eVar5.f1878d.n();
                        eVar4.f1895a.f1879e.m();
                        eVar4.f1897c = false;
                    } else {
                        i15 = 0;
                    }
                    eVar4.b(eVar4.f1898d);
                    e eVar6 = eVar4.f1895a;
                    eVar6.U = i15;
                    eVar6.V = i15;
                    int k = eVar6.k(i15);
                    int k2 = eVar4.f1895a.k(1);
                    if (eVar4.f1896b) {
                        eVar4.c();
                    }
                    int s = eVar4.f1895a.s();
                    int t = eVar4.f1895a.t();
                    eVar4.f1895a.f1878d.f1935h.c(s);
                    eVar4.f1895a.f1879e.f1935h.c(t);
                    eVar4.g();
                    if (k == 2 || k2 == 2) {
                        if (z15) {
                            Iterator<o> it2 = eVar4.f1899e.iterator();
                            while (true) {
                                if (it2.hasNext()) {
                                    if (!it2.next().k()) {
                                        z15 = false;
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                        if (z15 && k == 2) {
                            e eVar7 = eVar4.f1895a;
                            z4 = z3;
                            interfaceC0029b = interfaceC0029b4;
                            eVar7.O[0] = 1;
                            eVar7.M(eVar4.d(eVar7, 0));
                            e eVar8 = eVar4.f1895a;
                            eVar8.f1878d.f1932e.c(eVar8.r());
                        } else {
                            z4 = z3;
                            interfaceC0029b = interfaceC0029b4;
                        }
                        if (z15 && k2 == 2) {
                            e eVar9 = eVar4.f1895a;
                            eVar9.O[1] = 1;
                            eVar9.H(eVar4.d(eVar9, 1));
                            e eVar10 = eVar4.f1895a;
                            eVar10.f1879e.f1932e.c(eVar10.l());
                        }
                    } else {
                        z4 = z3;
                        interfaceC0029b = interfaceC0029b4;
                    }
                    e eVar11 = eVar4.f1895a;
                    int[] iArr = eVar11.O;
                    i5 = r;
                    if (iArr[0] == 1 || iArr[0] == 4) {
                        int r2 = eVar11.r() + s;
                        eVar4.f1895a.f1878d.i.c(r2);
                        eVar4.f1895a.f1878d.f1932e.c(r2 - s);
                        eVar4.g();
                        e eVar12 = eVar4.f1895a;
                        int[] iArr2 = eVar12.O;
                        if (iArr2[1] == 1 || iArr2[1] == 4) {
                            int l2 = eVar12.l() + t;
                            eVar4.f1895a.f1879e.i.c(l2);
                            eVar4.f1895a.f1879e.f1932e.c(l2 - t);
                        }
                        eVar4.g();
                        z11 = true;
                    } else {
                        z11 = false;
                    }
                    Iterator<o> it3 = eVar4.f1899e.iterator();
                    while (it3.hasNext()) {
                        o next2 = it3.next();
                        if (next2.f1929b != eVar4.f1895a || next2.f1934g) {
                            next2.e();
                        }
                    }
                    Iterator<o> it4 = eVar4.f1899e.iterator();
                    while (it4.hasNext()) {
                        o next3 = it4.next();
                        if (z11 || next3.f1929b != eVar4.f1895a) {
                            if (!next3.f1935h.j || ((!next3.i.j && !(next3 instanceof b.h.b.i.l.i)) || (!next3.f1932e.j && !(next3 instanceof b.h.b.i.l.c) && !(next3 instanceof b.h.b.i.l.i)))) {
                                z12 = false;
                                break;
                            }
                        }
                    }
                    z12 = true;
                    eVar4.f1895a.I(k);
                    eVar4.f1895a.L(k2);
                    z5 = z12;
                    i13 = 1073741824;
                    i6 = 2;
                } else {
                    z4 = z3;
                    interfaceC0029b = interfaceC0029b4;
                    i5 = r;
                    b.h.b.i.l.e eVar13 = eVar3.n0;
                    if (eVar13.f1896b) {
                        Iterator<b.h.b.i.d> it5 = eVar13.f1895a.l0.iterator();
                        while (it5.hasNext()) {
                            b.h.b.i.d next4 = it5.next();
                            next4.h();
                            next4.f1875a = false;
                            b.h.b.i.l.k kVar2 = next4.f1878d;
                            kVar2.f1932e.j = false;
                            kVar2.f1934g = false;
                            kVar2.n();
                            m mVar2 = next4.f1879e;
                            mVar2.f1932e.j = false;
                            mVar2.f1934g = false;
                            mVar2.m();
                        }
                        i12 = 0;
                        eVar13.f1895a.h();
                        e eVar14 = eVar13.f1895a;
                        eVar14.f1875a = false;
                        b.h.b.i.l.k kVar3 = eVar14.f1878d;
                        kVar3.f1932e.j = false;
                        kVar3.f1934g = false;
                        kVar3.n();
                        m mVar3 = eVar13.f1895a.f1879e;
                        mVar3.f1932e.j = false;
                        mVar3.f1934g = false;
                        mVar3.m();
                        eVar13.c();
                    } else {
                        i12 = 0;
                    }
                    eVar13.b(eVar13.f1898d);
                    e eVar15 = eVar13.f1895a;
                    eVar15.U = i12;
                    eVar15.V = i12;
                    eVar15.f1878d.f1935h.c(i12);
                    eVar13.f1895a.f1879e.f1935h.c(i12);
                    i13 = 1073741824;
                    if (mode == 1073741824) {
                        i14 = 1;
                        i6 = 1;
                        z10 = eVar3.V(b2, i12) & true;
                    } else {
                        i14 = 1;
                        z10 = true;
                        i6 = 0;
                    }
                    if (mode2 == 1073741824) {
                        z5 = eVar3.V(b2, i14) & z10;
                        i6++;
                    } else {
                        z5 = z10;
                    }
                }
                if (z5) {
                    eVar3.N(mode == i13, mode2 == i13);
                }
            }
            if (z5 || i6 != 2) {
                int i20 = eVar3.x0;
                if (size > 0) {
                    int size5 = eVar3.l0.size();
                    boolean Y = eVar3.Y(64);
                    b.InterfaceC0029b interfaceC0029b5 = eVar3.o0;
                    for (int i21 = 0; i21 < size5; i21++) {
                        b.h.b.i.d dVar2 = eVar3.l0.get(i21);
                        if (!(dVar2 instanceof b.h.b.i.f) && !(dVar2 instanceof b.h.b.i.a) && !dVar2.A && (!Y || (kVar = dVar2.f1878d) == null || (mVar = dVar2.f1879e) == null || !kVar.f1932e.j || !mVar.f1932e.j)) {
                            int k3 = dVar2.k(0);
                            int k4 = dVar2.k(1);
                            boolean z16 = k3 == 3 && dVar2.l != 1 && k4 == 3 && dVar2.m != 1;
                            if (!z16 && eVar3.Y(1) && !(dVar2 instanceof b.h.b.i.j)) {
                                if (k3 == 3 && dVar2.l == 0 && k4 != 3 && !dVar2.w()) {
                                    z16 = true;
                                }
                                if (k4 == 3 && dVar2.m == 0 && k3 != 3 && !dVar2.w()) {
                                    z16 = true;
                                }
                                if ((k3 == 3 || k4 == 3) && dVar2.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    z16 = true;
                                }
                            }
                            if (!z16) {
                                bVar.a(interfaceC0029b5, dVar2, 0);
                            }
                        }
                    }
                    b bVar3 = (b) interfaceC0029b5;
                    int childCount = bVar3.f206a.getChildCount();
                    for (int i22 = 0; i22 < childCount; i22++) {
                        View childAt = bVar3.f206a.getChildAt(i22);
                        if (childAt instanceof g) {
                            g gVar = (g) childAt;
                            if (gVar.f2007c != null) {
                                a aVar7 = (a) gVar.getLayoutParams();
                                a aVar8 = (a) gVar.f2007c.getLayoutParams();
                                aVar8.l0.c0 = 0;
                                if (aVar7.l0.m() != 1) {
                                    aVar7.l0.M(aVar8.l0.r());
                                }
                                if (aVar7.l0.q() != 1) {
                                    aVar7.l0.H(aVar8.l0.l());
                                }
                                aVar8.l0.c0 = 8;
                            }
                        }
                    }
                    int size6 = bVar3.f206a.mConstraintHelpers.size();
                    if (size6 > 0) {
                        for (int i23 = 0; i23 < size6; i23++) {
                            ((b.h.c.b) bVar3.f206a.mConstraintHelpers.get(i23)).i();
                        }
                    }
                }
                bVar.c(eVar3);
                size2 = bVar.f1884a.size();
                int i24 = i5;
                if (size > 0) {
                    bVar.b(eVar3, i24, l);
                }
                if (size2 <= 0) {
                    boolean z17 = eVar.m() == 2;
                    boolean z18 = eVar.q() == 2;
                    int max5 = Math.max(eVar.r(), bVar.f1886c.X);
                    int max6 = Math.max(eVar.l(), bVar.f1886c.Y);
                    int i25 = 0;
                    boolean z19 = false;
                    while (i25 < size2) {
                        b.h.b.i.d dVar3 = bVar.f1884a.get(i25);
                        if (dVar3 instanceof b.h.b.i.j) {
                            int r3 = dVar3.r();
                            int l3 = dVar3.l();
                            i11 = i20;
                            interfaceC0029b3 = interfaceC0029b;
                            boolean a2 = z19 | bVar.a(interfaceC0029b3, dVar3, 1);
                            int r4 = dVar3.r();
                            int l4 = dVar3.l();
                            if (r4 != r3) {
                                dVar3.M(r4);
                                if (!z17 || dVar3.p() <= max5) {
                                    aVar3 = aVar2;
                                } else {
                                    aVar3 = aVar2;
                                    max5 = Math.max(max5, dVar3.i(aVar3).d() + dVar3.p());
                                }
                                z9 = true;
                            } else {
                                aVar3 = aVar2;
                                z9 = a2;
                            }
                            if (l4 != l3) {
                                dVar3.H(l4);
                                if (!z18 || dVar3.j() <= max6) {
                                    aVar4 = aVar;
                                } else {
                                    aVar4 = aVar;
                                    max6 = Math.max(max6, dVar3.i(aVar4).d() + dVar3.j());
                                }
                                z9 = true;
                            } else {
                                aVar4 = aVar;
                            }
                            b.h.b.i.j jVar = (b.h.b.i.j) dVar3;
                            z19 = z9 | false;
                        } else {
                            i11 = i20;
                            aVar4 = aVar;
                            aVar3 = aVar2;
                            interfaceC0029b3 = interfaceC0029b;
                        }
                        i25++;
                        interfaceC0029b = interfaceC0029b3;
                        aVar2 = aVar3;
                        aVar = aVar4;
                        i20 = i11;
                    }
                    i7 = i20;
                    c.a aVar9 = aVar;
                    c.a aVar10 = aVar2;
                    b.InterfaceC0029b interfaceC0029b6 = interfaceC0029b;
                    int i26 = 0;
                    int i27 = 0;
                    while (true) {
                        if (i27 >= 2) {
                            i8 = i24;
                            eVar2 = eVar3;
                            z6 = true;
                            break;
                        }
                        boolean z20 = z19;
                        int i28 = i26;
                        while (i28 < size2) {
                            b.h.b.i.d dVar4 = bVar.f1884a.get(i28);
                            if ((!(dVar4 instanceof b.h.b.i.g) || (dVar4 instanceof b.h.b.i.j)) && !(dVar4 instanceof b.h.b.i.f)) {
                                i9 = size2;
                                if (dVar4.c0 != 8 && ((!z4 || !dVar4.f1878d.f1932e.j || !dVar4.f1879e.f1932e.j) && !(dVar4 instanceof b.h.b.i.j))) {
                                    int r5 = dVar4.r();
                                    int l5 = dVar4.l();
                                    i10 = i24;
                                    int i29 = dVar4.W;
                                    int i30 = i27 == 1 ? 2 : 1;
                                    interfaceC0029b2 = interfaceC0029b6;
                                    int r6 = dVar4.r();
                                    z20 = bVar.a(interfaceC0029b6, dVar4, i30) | z20;
                                    int l6 = dVar4.l();
                                    if (r6 != r5) {
                                        dVar4.M(r6);
                                        if (z17 && dVar4.p() > max5) {
                                            max5 = Math.max(max5, dVar4.i(aVar10).d() + dVar4.p());
                                        }
                                        z20 = true;
                                    }
                                    if (l6 != l5) {
                                        dVar4.H(l6);
                                        if (z18 && dVar4.j() > max6) {
                                            max6 = Math.max(max6, dVar4.i(aVar9).d() + dVar4.j());
                                        }
                                        z20 = true;
                                    }
                                    if (dVar4.y && i29 != dVar4.W) {
                                        z20 = true;
                                    }
                                    i28++;
                                    size2 = i9;
                                    interfaceC0029b6 = interfaceC0029b2;
                                    i24 = i10;
                                }
                            } else {
                                i9 = size2;
                            }
                            interfaceC0029b2 = interfaceC0029b6;
                            i10 = i24;
                            i28++;
                            size2 = i9;
                            interfaceC0029b6 = interfaceC0029b2;
                            i24 = i10;
                        }
                        b.InterfaceC0029b interfaceC0029b7 = interfaceC0029b6;
                        int i31 = size2;
                        int i32 = i24;
                        if (!z20) {
                            eVar2 = eVar;
                            i8 = i32;
                            z6 = true;
                            z19 = z20;
                            break;
                        }
                        bVar.b(eVar, i32, l);
                        i27++;
                        size2 = i31;
                        eVar3 = eVar;
                        i24 = i32;
                        interfaceC0029b6 = interfaceC0029b7;
                        i26 = 0;
                        z19 = false;
                    }
                    if (z19) {
                        bVar.b(eVar2, i8, l);
                        if (eVar.r() < max5) {
                            eVar2.M(max5);
                            z7 = z6;
                        } else {
                            z7 = false;
                        }
                        if (eVar.l() < max6) {
                            eVar2.H(max6);
                            z8 = z6;
                        } else {
                            z8 = z7;
                        }
                        if (z8) {
                            bVar.b(eVar2, i8, l);
                        }
                    }
                } else {
                    i7 = i20;
                    eVar2 = eVar3;
                }
                eVar2.Z(i7);
            }
            return;
        }
        i4 = max3;
        int i172 = size3 - paddingWidth;
        int i182 = size4 - i16;
        setSelfDimensionBehaviour(eVar, mode, i172, mode2, i182);
        eVar3.r0 = i4;
        eVar3.s0 = max;
        bVar = eVar3.m0;
        Objects.requireNonNull(bVar);
        c.a aVar52 = c.a.BOTTOM;
        c.a aVar62 = c.a.RIGHT;
        b.InterfaceC0029b interfaceC0029b42 = eVar3.o0;
        size = eVar3.l0.size();
        int r7 = eVar.r();
        int l7 = eVar.l();
        boolean b22 = i.b(i, 128);
        if (b22) {
        }
        if (z) {
        }
        aVar = aVar52;
        aVar2 = aVar62;
        z2 = z;
        z3 = z2 & ((mode != 1073741824 && mode2 == 1073741824) || b22);
        if (z3) {
        }
        if (z5) {
        }
        int i202 = eVar3.x0;
        if (size > 0) {
        }
        bVar.c(eVar3);
        size2 = bVar.f1884a.size();
        int i242 = i5;
        if (size > 0) {
        }
        if (size2 <= 0) {
        }
        eVar2.Z(i7);
    }

    public void setConstraintSet(d dVar) {
        this.mConstraintSet = dVar;
    }

    public void setDesignInformation(int i, Object obj, Object obj2) {
        if (i == 0 && (obj instanceof String) && (obj2 instanceof Integer)) {
            if (this.mDesignIds == null) {
                this.mDesignIds = new HashMap<>();
            }
            String str = (String) obj;
            int indexOf = str.indexOf("/");
            if (indexOf != -1) {
                str = str.substring(indexOf + 1);
            }
            this.mDesignIds.put(str, Integer.valueOf(((Integer) obj2).intValue()));
        }
    }

    @Override // android.view.View
    public void setId(int i) {
        this.mChildrenByIds.remove(getId());
        super.setId(i);
        this.mChildrenByIds.put(getId(), this);
    }

    public void setMaxHeight(int i) {
        if (i == this.mMaxHeight) {
            return;
        }
        this.mMaxHeight = i;
        requestLayout();
    }

    public void setMaxWidth(int i) {
        if (i == this.mMaxWidth) {
            return;
        }
        this.mMaxWidth = i;
        requestLayout();
    }

    public void setMinHeight(int i) {
        if (i == this.mMinHeight) {
            return;
        }
        this.mMinHeight = i;
        requestLayout();
    }

    public void setMinWidth(int i) {
        if (i == this.mMinWidth) {
            return;
        }
        this.mMinWidth = i;
        requestLayout();
    }

    public void setOnConstraintsChanged(f fVar) {
        b.h.c.c cVar = this.mConstraintLayoutSpec;
        if (cVar != null) {
            Objects.requireNonNull(cVar);
        }
    }

    public void setOptimizationLevel(int i) {
        this.mOptimizationLevel = i;
        e eVar = this.mLayoutWidget;
        eVar.x0 = i;
        b.h.b.d.f1829a = eVar.Y(512);
    }

    /* JADX WARN: Removed duplicated region for block: B:16:0x003b  */
    /* JADX WARN: Removed duplicated region for block: B:23:0x0055  */
    /* JADX WARN: Removed duplicated region for block: B:27:0x0063  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void setSelfDimensionBehaviour(e eVar, int i, int i2, int i3, int i4) {
        int i5;
        int i6;
        int i7;
        b bVar = this.mMeasurer;
        int i8 = bVar.f210e;
        int i9 = bVar.f209d;
        int childCount = getChildCount();
        int i10 = 2;
        if (i != Integer.MIN_VALUE) {
            if (i != 0) {
                if (i == 1073741824) {
                    i5 = Math.min(this.mMaxWidth - i9, i2);
                    i6 = 1;
                    if (i3 != Integer.MIN_VALUE) {
                        if (i3 == 0) {
                            if (childCount == 0) {
                                i4 = Math.max(0, this.mMinHeight);
                            }
                            i4 = 0;
                        } else if (i3 != 1073741824) {
                            i10 = 1;
                            i4 = 0;
                        } else {
                            i4 = Math.min(this.mMaxHeight - i8, i4);
                            i10 = 1;
                        }
                    } else if (childCount == 0) {
                        i4 = Math.max(0, this.mMinHeight);
                    }
                    if (i5 == eVar.r() || i4 != eVar.l()) {
                        eVar.n0.f1897c = true;
                    }
                    eVar.U = 0;
                    eVar.V = 0;
                    int[] iArr = eVar.w;
                    iArr[0] = this.mMaxWidth - i9;
                    iArr[1] = this.mMaxHeight - i8;
                    eVar.K(0);
                    eVar.J(0);
                    eVar.O[0] = i6;
                    eVar.M(i5);
                    eVar.O[1] = i10;
                    eVar.H(i4);
                    eVar.K(this.mMinWidth - i9);
                    eVar.J(this.mMinHeight - i8);
                }
                i7 = 1;
            } else if (childCount == 0) {
                i2 = Math.max(0, this.mMinWidth);
            } else {
                i7 = 2;
            }
            i6 = i7;
            i5 = 0;
            if (i3 != Integer.MIN_VALUE) {
            }
            if (i5 == eVar.r()) {
            }
            eVar.n0.f1897c = true;
            eVar.U = 0;
            eVar.V = 0;
            int[] iArr2 = eVar.w;
            iArr2[0] = this.mMaxWidth - i9;
            iArr2[1] = this.mMaxHeight - i8;
            eVar.K(0);
            eVar.J(0);
            eVar.O[0] = i6;
            eVar.M(i5);
            eVar.O[1] = i10;
            eVar.H(i4);
            eVar.K(this.mMinWidth - i9);
            eVar.J(this.mMinHeight - i8);
        } else if (childCount == 0) {
            i2 = Math.max(0, this.mMinWidth);
        }
        i5 = i2;
        i6 = 2;
        if (i3 != Integer.MIN_VALUE) {
        }
        if (i5 == eVar.r()) {
        }
        eVar.n0.f1897c = true;
        eVar.U = 0;
        eVar.V = 0;
        int[] iArr22 = eVar.w;
        iArr22[0] = this.mMaxWidth - i9;
        iArr22[1] = this.mMaxHeight - i8;
        eVar.K(0);
        eVar.J(0);
        eVar.O[0] = i6;
        eVar.M(i5);
        eVar.O[1] = i10;
        eVar.H(i4);
        eVar.K(this.mMinWidth - i9);
        eVar.J(this.mMinHeight - i8);
    }

    public void setState(int i, int i2, int i3) {
        d dVar;
        c.a aVar;
        int a2;
        b.h.c.c cVar = this.mConstraintLayoutSpec;
        if (cVar != null) {
            float f2 = i2;
            float f3 = i3;
            int i4 = cVar.f1951b;
            if (i4 == i) {
                if (i == -1) {
                    aVar = cVar.f1953d.valueAt(0);
                } else {
                    aVar = cVar.f1953d.get(i4);
                }
                int i5 = cVar.f1952c;
                if ((i5 == -1 || !aVar.f1956b.get(i5).a(f2, f3)) && cVar.f1952c != (a2 = aVar.a(f2, f3))) {
                    d dVar2 = a2 == -1 ? null : aVar.f1956b.get(a2).f1964f;
                    if (a2 != -1) {
                        int i6 = aVar.f1956b.get(a2).f1963e;
                    }
                    if (dVar2 == null) {
                        return;
                    }
                    cVar.f1952c = a2;
                    dVar2.a(cVar.f1950a);
                    return;
                }
                return;
            }
            cVar.f1951b = i;
            c.a aVar2 = cVar.f1953d.get(i);
            int a3 = aVar2.a(f2, f3);
            if (a3 == -1) {
                dVar = aVar2.f1958d;
            } else {
                dVar = aVar2.f1956b.get(a3).f1964f;
            }
            if (a3 != -1) {
                int i7 = aVar2.f1956b.get(a3).f1963e;
            }
            if (dVar == null) {
                Log.v("ConstraintLayoutStates", "NO Constraint set found ! id=" + i + ", dim =" + f2 + ", " + f3);
                return;
            }
            cVar.f1952c = a3;
            dVar.a(cVar.f1950a);
        }
    }

    @Override // android.view.ViewGroup
    public boolean shouldDelayChildPressedState() {
        return false;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.view.ViewGroup
    public a generateDefaultLayoutParams() {
        return new a(-2, -2);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.view.ViewGroup
    public a generateLayoutParams(AttributeSet attributeSet) {
        return new a(getContext(), attributeSet);
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return new a(layoutParams);
    }

    public ConstraintLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mChildrenByIds = new SparseArray<>();
        this.mConstraintHelpers = new ArrayList<>(4);
        this.mLayoutWidget = new e();
        this.mMinWidth = 0;
        this.mMinHeight = 0;
        this.mMaxWidth = Integer.MAX_VALUE;
        this.mMaxHeight = Integer.MAX_VALUE;
        this.mDirtyHierarchy = true;
        this.mOptimizationLevel = Imgcodecs.IMWRITE_TIFF_XDPI;
        this.mConstraintSet = null;
        this.mConstraintLayoutSpec = null;
        this.mConstraintSetId = -1;
        this.mDesignIds = new HashMap<>();
        this.mLastMeasureWidth = -1;
        this.mLastMeasureHeight = -1;
        this.mLastMeasureWidthSize = -1;
        this.mLastMeasureHeightSize = -1;
        this.mLastMeasureWidthMode = 0;
        this.mLastMeasureHeightMode = 0;
        this.mTempMapIdToWidget = new SparseArray<>();
        this.mMeasurer = new b(this);
        this.mOnMeasureWidthMeasureSpec = 0;
        this.mOnMeasureHeightMeasureSpec = 0;
        init(attributeSet, 0, 0);
    }

    public ConstraintLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mChildrenByIds = new SparseArray<>();
        this.mConstraintHelpers = new ArrayList<>(4);
        this.mLayoutWidget = new e();
        this.mMinWidth = 0;
        this.mMinHeight = 0;
        this.mMaxWidth = Integer.MAX_VALUE;
        this.mMaxHeight = Integer.MAX_VALUE;
        this.mDirtyHierarchy = true;
        this.mOptimizationLevel = Imgcodecs.IMWRITE_TIFF_XDPI;
        this.mConstraintSet = null;
        this.mConstraintLayoutSpec = null;
        this.mConstraintSetId = -1;
        this.mDesignIds = new HashMap<>();
        this.mLastMeasureWidth = -1;
        this.mLastMeasureHeight = -1;
        this.mLastMeasureWidthSize = -1;
        this.mLastMeasureHeightSize = -1;
        this.mLastMeasureWidthMode = 0;
        this.mLastMeasureHeightMode = 0;
        this.mTempMapIdToWidget = new SparseArray<>();
        this.mMeasurer = new b(this);
        this.mOnMeasureWidthMeasureSpec = 0;
        this.mOnMeasureHeightMeasureSpec = 0;
        init(attributeSet, i, 0);
    }

    @TargetApi(21)
    public ConstraintLayout(Context context, AttributeSet attributeSet, int i, int i2) {
        super(context, attributeSet, i, i2);
        this.mChildrenByIds = new SparseArray<>();
        this.mConstraintHelpers = new ArrayList<>(4);
        this.mLayoutWidget = new e();
        this.mMinWidth = 0;
        this.mMinHeight = 0;
        this.mMaxWidth = Integer.MAX_VALUE;
        this.mMaxHeight = Integer.MAX_VALUE;
        this.mDirtyHierarchy = true;
        this.mOptimizationLevel = Imgcodecs.IMWRITE_TIFF_XDPI;
        this.mConstraintSet = null;
        this.mConstraintLayoutSpec = null;
        this.mConstraintSetId = -1;
        this.mDesignIds = new HashMap<>();
        this.mLastMeasureWidth = -1;
        this.mLastMeasureHeight = -1;
        this.mLastMeasureWidthSize = -1;
        this.mLastMeasureHeightSize = -1;
        this.mLastMeasureWidthMode = 0;
        this.mLastMeasureHeightMode = 0;
        this.mTempMapIdToWidget = new SparseArray<>();
        this.mMeasurer = new b(this);
        this.mOnMeasureWidthMeasureSpec = 0;
        this.mOnMeasureHeightMeasureSpec = 0;
        init(attributeSet, i, i2);
    }

    /* loaded from: classes.dex */
    public static class a extends ViewGroup.MarginLayoutParams {
        public float A;
        public String B;
        public int C;
        public float D;
        public float E;
        public int F;
        public int G;
        public int H;
        public int I;
        public int J;
        public int K;
        public int L;
        public int M;
        public float N;
        public float O;
        public int P;
        public int Q;
        public int R;
        public boolean S;
        public boolean T;
        public String U;
        public boolean V;
        public boolean W;
        public boolean X;
        public boolean Y;
        public boolean Z;

        /* renamed from: a  reason: collision with root package name */
        public int f197a;
        public boolean a0;

        /* renamed from: b  reason: collision with root package name */
        public int f198b;
        public int b0;

        /* renamed from: c  reason: collision with root package name */
        public float f199c;
        public int c0;

        /* renamed from: d  reason: collision with root package name */
        public int f200d;
        public int d0;

        /* renamed from: e  reason: collision with root package name */
        public int f201e;
        public int e0;

        /* renamed from: f  reason: collision with root package name */
        public int f202f;
        public int f0;

        /* renamed from: g  reason: collision with root package name */
        public int f203g;
        public int g0;

        /* renamed from: h  reason: collision with root package name */
        public int f204h;
        public float h0;
        public int i;
        public int i0;
        public int j;
        public int j0;
        public int k;
        public float k0;
        public int l;
        public b.h.b.i.d l0;
        public int m;
        public int n;
        public float o;
        public int p;
        public int q;
        public int r;
        public int s;
        public int t;
        public int u;
        public int v;
        public int w;
        public int x;
        public int y;
        public float z;

        /* renamed from: androidx.constraintlayout.widget.ConstraintLayout$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public static class C0003a {

            /* renamed from: a  reason: collision with root package name */
            public static final SparseIntArray f205a;

            static {
                SparseIntArray sparseIntArray = new SparseIntArray();
                f205a = sparseIntArray;
                sparseIntArray.append(64, 8);
                sparseIntArray.append(65, 9);
                sparseIntArray.append(67, 10);
                sparseIntArray.append(68, 11);
                sparseIntArray.append(74, 12);
                sparseIntArray.append(73, 13);
                sparseIntArray.append(46, 14);
                sparseIntArray.append(45, 15);
                sparseIntArray.append(43, 16);
                sparseIntArray.append(47, 2);
                sparseIntArray.append(49, 3);
                sparseIntArray.append(48, 4);
                sparseIntArray.append(82, 49);
                sparseIntArray.append(83, 50);
                sparseIntArray.append(53, 5);
                sparseIntArray.append(54, 6);
                sparseIntArray.append(55, 7);
                sparseIntArray.append(0, 1);
                sparseIntArray.append(69, 17);
                sparseIntArray.append(70, 18);
                sparseIntArray.append(52, 19);
                sparseIntArray.append(51, 20);
                sparseIntArray.append(86, 21);
                sparseIntArray.append(89, 22);
                sparseIntArray.append(87, 23);
                sparseIntArray.append(84, 24);
                sparseIntArray.append(88, 25);
                sparseIntArray.append(85, 26);
                sparseIntArray.append(60, 29);
                sparseIntArray.append(75, 30);
                sparseIntArray.append(50, 44);
                sparseIntArray.append(62, 45);
                sparseIntArray.append(77, 46);
                sparseIntArray.append(61, 47);
                sparseIntArray.append(76, 48);
                sparseIntArray.append(41, 27);
                sparseIntArray.append(40, 28);
                sparseIntArray.append(78, 31);
                sparseIntArray.append(56, 32);
                sparseIntArray.append(80, 33);
                sparseIntArray.append(79, 34);
                sparseIntArray.append(81, 35);
                sparseIntArray.append(58, 36);
                sparseIntArray.append(57, 37);
                sparseIntArray.append(59, 38);
                sparseIntArray.append(63, 39);
                sparseIntArray.append(72, 40);
                sparseIntArray.append(66, 41);
                sparseIntArray.append(44, 42);
                sparseIntArray.append(42, 43);
                sparseIntArray.append(71, 51);
            }
        }

        public a(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            int i;
            this.f197a = -1;
            this.f198b = -1;
            this.f199c = -1.0f;
            this.f200d = -1;
            this.f201e = -1;
            this.f202f = -1;
            this.f203g = -1;
            this.f204h = -1;
            this.i = -1;
            this.j = -1;
            this.k = -1;
            this.l = -1;
            this.m = -1;
            this.n = 0;
            this.o = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.p = -1;
            this.q = -1;
            this.r = -1;
            this.s = -1;
            this.t = -1;
            this.u = -1;
            this.v = -1;
            this.w = -1;
            this.x = -1;
            this.y = -1;
            this.z = 0.5f;
            this.A = 0.5f;
            this.B = null;
            this.C = 1;
            this.D = -1.0f;
            this.E = -1.0f;
            this.F = 0;
            this.G = 0;
            this.H = 0;
            this.I = 0;
            this.J = 0;
            this.K = 0;
            this.L = 0;
            this.M = 0;
            this.N = 1.0f;
            this.O = 1.0f;
            this.P = -1;
            this.Q = -1;
            this.R = -1;
            this.S = false;
            this.T = false;
            this.U = null;
            this.V = true;
            this.W = true;
            this.X = false;
            this.Y = false;
            this.Z = false;
            this.a0 = false;
            this.b0 = -1;
            this.c0 = -1;
            this.d0 = -1;
            this.e0 = -1;
            this.f0 = -1;
            this.g0 = -1;
            this.h0 = 0.5f;
            this.l0 = new b.h.b.i.d();
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.h.c.i.f2010b);
            int indexCount = obtainStyledAttributes.getIndexCount();
            for (int i2 = 0; i2 < indexCount; i2++) {
                int index = obtainStyledAttributes.getIndex(i2);
                int i3 = C0003a.f205a.get(index);
                switch (i3) {
                    case 1:
                        this.R = obtainStyledAttributes.getInt(index, this.R);
                        break;
                    case 2:
                        int resourceId = obtainStyledAttributes.getResourceId(index, this.m);
                        this.m = resourceId;
                        if (resourceId == -1) {
                            this.m = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 3:
                        this.n = obtainStyledAttributes.getDimensionPixelSize(index, this.n);
                        break;
                    case 4:
                        float f2 = obtainStyledAttributes.getFloat(index, this.o) % 360.0f;
                        this.o = f2;
                        if (f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            this.o = (360.0f - f2) % 360.0f;
                            break;
                        } else {
                            break;
                        }
                    case 5:
                        this.f197a = obtainStyledAttributes.getDimensionPixelOffset(index, this.f197a);
                        break;
                    case 6:
                        this.f198b = obtainStyledAttributes.getDimensionPixelOffset(index, this.f198b);
                        break;
                    case 7:
                        this.f199c = obtainStyledAttributes.getFloat(index, this.f199c);
                        break;
                    case 8:
                        int resourceId2 = obtainStyledAttributes.getResourceId(index, this.f200d);
                        this.f200d = resourceId2;
                        if (resourceId2 == -1) {
                            this.f200d = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 9:
                        int resourceId3 = obtainStyledAttributes.getResourceId(index, this.f201e);
                        this.f201e = resourceId3;
                        if (resourceId3 == -1) {
                            this.f201e = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 10:
                        int resourceId4 = obtainStyledAttributes.getResourceId(index, this.f202f);
                        this.f202f = resourceId4;
                        if (resourceId4 == -1) {
                            this.f202f = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 11:
                        int resourceId5 = obtainStyledAttributes.getResourceId(index, this.f203g);
                        this.f203g = resourceId5;
                        if (resourceId5 == -1) {
                            this.f203g = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 12:
                        int resourceId6 = obtainStyledAttributes.getResourceId(index, this.f204h);
                        this.f204h = resourceId6;
                        if (resourceId6 == -1) {
                            this.f204h = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 13:
                        int resourceId7 = obtainStyledAttributes.getResourceId(index, this.i);
                        this.i = resourceId7;
                        if (resourceId7 == -1) {
                            this.i = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 14:
                        int resourceId8 = obtainStyledAttributes.getResourceId(index, this.j);
                        this.j = resourceId8;
                        if (resourceId8 == -1) {
                            this.j = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 15:
                        int resourceId9 = obtainStyledAttributes.getResourceId(index, this.k);
                        this.k = resourceId9;
                        if (resourceId9 == -1) {
                            this.k = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 16:
                        int resourceId10 = obtainStyledAttributes.getResourceId(index, this.l);
                        this.l = resourceId10;
                        if (resourceId10 == -1) {
                            this.l = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 17:
                        int resourceId11 = obtainStyledAttributes.getResourceId(index, this.p);
                        this.p = resourceId11;
                        if (resourceId11 == -1) {
                            this.p = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 18:
                        int resourceId12 = obtainStyledAttributes.getResourceId(index, this.q);
                        this.q = resourceId12;
                        if (resourceId12 == -1) {
                            this.q = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 19:
                        int resourceId13 = obtainStyledAttributes.getResourceId(index, this.r);
                        this.r = resourceId13;
                        if (resourceId13 == -1) {
                            this.r = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 20:
                        int resourceId14 = obtainStyledAttributes.getResourceId(index, this.s);
                        this.s = resourceId14;
                        if (resourceId14 == -1) {
                            this.s = obtainStyledAttributes.getInt(index, -1);
                            break;
                        } else {
                            break;
                        }
                    case 21:
                        this.t = obtainStyledAttributes.getDimensionPixelSize(index, this.t);
                        break;
                    case 22:
                        this.u = obtainStyledAttributes.getDimensionPixelSize(index, this.u);
                        break;
                    case 23:
                        this.v = obtainStyledAttributes.getDimensionPixelSize(index, this.v);
                        break;
                    case 24:
                        this.w = obtainStyledAttributes.getDimensionPixelSize(index, this.w);
                        break;
                    case 25:
                        this.x = obtainStyledAttributes.getDimensionPixelSize(index, this.x);
                        break;
                    case 26:
                        this.y = obtainStyledAttributes.getDimensionPixelSize(index, this.y);
                        break;
                    case 27:
                        this.S = obtainStyledAttributes.getBoolean(index, this.S);
                        break;
                    case 28:
                        this.T = obtainStyledAttributes.getBoolean(index, this.T);
                        break;
                    case 29:
                        this.z = obtainStyledAttributes.getFloat(index, this.z);
                        break;
                    case 30:
                        this.A = obtainStyledAttributes.getFloat(index, this.A);
                        break;
                    case 31:
                        int i4 = obtainStyledAttributes.getInt(index, 0);
                        this.H = i4;
                        if (i4 == 1) {
                            Log.e(ConstraintLayout.TAG, "layout_constraintWidth_default=\"wrap\" is deprecated.\nUse layout_width=\"WRAP_CONTENT\" and layout_constrainedWidth=\"true\" instead.");
                            break;
                        } else {
                            break;
                        }
                    case 32:
                        int i5 = obtainStyledAttributes.getInt(index, 0);
                        this.I = i5;
                        if (i5 == 1) {
                            Log.e(ConstraintLayout.TAG, "layout_constraintHeight_default=\"wrap\" is deprecated.\nUse layout_height=\"WRAP_CONTENT\" and layout_constrainedHeight=\"true\" instead.");
                            break;
                        } else {
                            break;
                        }
                    case 33:
                        try {
                            this.J = obtainStyledAttributes.getDimensionPixelSize(index, this.J);
                            break;
                        } catch (Exception unused) {
                            if (obtainStyledAttributes.getInt(index, this.J) == -2) {
                                this.J = -2;
                                break;
                            } else {
                                break;
                            }
                        }
                    case 34:
                        try {
                            this.L = obtainStyledAttributes.getDimensionPixelSize(index, this.L);
                            break;
                        } catch (Exception unused2) {
                            if (obtainStyledAttributes.getInt(index, this.L) == -2) {
                                this.L = -2;
                                break;
                            } else {
                                break;
                            }
                        }
                    case 35:
                        this.N = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, obtainStyledAttributes.getFloat(index, this.N));
                        this.H = 2;
                        break;
                    case 36:
                        try {
                            this.K = obtainStyledAttributes.getDimensionPixelSize(index, this.K);
                            break;
                        } catch (Exception unused3) {
                            if (obtainStyledAttributes.getInt(index, this.K) == -2) {
                                this.K = -2;
                                break;
                            } else {
                                break;
                            }
                        }
                    case 37:
                        try {
                            this.M = obtainStyledAttributes.getDimensionPixelSize(index, this.M);
                            break;
                        } catch (Exception unused4) {
                            if (obtainStyledAttributes.getInt(index, this.M) == -2) {
                                this.M = -2;
                                break;
                            } else {
                                break;
                            }
                        }
                    case 38:
                        this.O = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, obtainStyledAttributes.getFloat(index, this.O));
                        this.I = 2;
                        break;
                    default:
                        switch (i3) {
                            case 44:
                                String string = obtainStyledAttributes.getString(index);
                                this.B = string;
                                this.C = -1;
                                if (string == null) {
                                    break;
                                } else {
                                    int length = string.length();
                                    int indexOf = this.B.indexOf(44);
                                    if (indexOf <= 0 || indexOf >= length - 1) {
                                        i = 0;
                                    } else {
                                        String substring = this.B.substring(0, indexOf);
                                        if (substring.equalsIgnoreCase("W")) {
                                            this.C = 0;
                                        } else if (substring.equalsIgnoreCase("H")) {
                                            this.C = 1;
                                        }
                                        i = indexOf + 1;
                                    }
                                    int indexOf2 = this.B.indexOf(58);
                                    if (indexOf2 >= 0 && indexOf2 < length - 1) {
                                        String substring2 = this.B.substring(i, indexOf2);
                                        String substring3 = this.B.substring(indexOf2 + 1);
                                        if (substring2.length() > 0 && substring3.length() > 0) {
                                            try {
                                                float parseFloat = Float.parseFloat(substring2);
                                                float parseFloat2 = Float.parseFloat(substring3);
                                                if (parseFloat > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && parseFloat2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                                    if (this.C == 1) {
                                                        Math.abs(parseFloat2 / parseFloat);
                                                        break;
                                                    } else {
                                                        Math.abs(parseFloat / parseFloat2);
                                                        break;
                                                    }
                                                }
                                            } catch (NumberFormatException unused5) {
                                                break;
                                            }
                                        }
                                    } else {
                                        String substring4 = this.B.substring(i);
                                        if (substring4.length() <= 0) {
                                            break;
                                        } else {
                                            Float.parseFloat(substring4);
                                            continue;
                                        }
                                    }
                                }
                                break;
                            case 45:
                                this.D = obtainStyledAttributes.getFloat(index, this.D);
                                continue;
                            case 46:
                                this.E = obtainStyledAttributes.getFloat(index, this.E);
                                continue;
                            case 47:
                                this.F = obtainStyledAttributes.getInt(index, 0);
                                continue;
                            case 48:
                                this.G = obtainStyledAttributes.getInt(index, 0);
                                continue;
                            case 49:
                                this.P = obtainStyledAttributes.getDimensionPixelOffset(index, this.P);
                                continue;
                            case 50:
                                this.Q = obtainStyledAttributes.getDimensionPixelOffset(index, this.Q);
                                continue;
                            case 51:
                                this.U = obtainStyledAttributes.getString(index);
                                continue;
                        }
                }
            }
            obtainStyledAttributes.recycle();
            a();
        }

        public void a() {
            this.Y = false;
            this.V = true;
            this.W = true;
            int i = ((ViewGroup.MarginLayoutParams) this).width;
            if (i == -2 && this.S) {
                this.V = false;
                if (this.H == 0) {
                    this.H = 1;
                }
            }
            int i2 = ((ViewGroup.MarginLayoutParams) this).height;
            if (i2 == -2 && this.T) {
                this.W = false;
                if (this.I == 0) {
                    this.I = 1;
                }
            }
            if (i == 0 || i == -1) {
                this.V = false;
                if (i == 0 && this.H == 1) {
                    ((ViewGroup.MarginLayoutParams) this).width = -2;
                    this.S = true;
                }
            }
            if (i2 == 0 || i2 == -1) {
                this.W = false;
                if (i2 == 0 && this.I == 1) {
                    ((ViewGroup.MarginLayoutParams) this).height = -2;
                    this.T = true;
                }
            }
            if (this.f199c == -1.0f && this.f197a == -1 && this.f198b == -1) {
                return;
            }
            this.Y = true;
            this.V = true;
            this.W = true;
            if (!(this.l0 instanceof b.h.b.i.f)) {
                this.l0 = new b.h.b.i.f();
            }
            ((b.h.b.i.f) this.l0).Q(this.R);
        }

        /* JADX WARN: Removed duplicated region for block: B:17:0x004c  */
        /* JADX WARN: Removed duplicated region for block: B:20:0x0053  */
        /* JADX WARN: Removed duplicated region for block: B:23:0x005a  */
        /* JADX WARN: Removed duplicated region for block: B:26:0x0060  */
        /* JADX WARN: Removed duplicated region for block: B:29:0x0066  */
        /* JADX WARN: Removed duplicated region for block: B:36:0x0078  */
        /* JADX WARN: Removed duplicated region for block: B:37:0x0080  */
        @Override // android.view.ViewGroup.MarginLayoutParams, android.view.ViewGroup.LayoutParams
        @TargetApi(17)
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void resolveLayoutDirection(int i) {
            int i2;
            int i3;
            int i4;
            int i5;
            int i6 = ((ViewGroup.MarginLayoutParams) this).leftMargin;
            int i7 = ((ViewGroup.MarginLayoutParams) this).rightMargin;
            super.resolveLayoutDirection(i);
            boolean z = false;
            boolean z2 = 1 == getLayoutDirection();
            this.d0 = -1;
            this.e0 = -1;
            this.b0 = -1;
            this.c0 = -1;
            this.f0 = -1;
            this.g0 = -1;
            this.f0 = this.t;
            this.g0 = this.v;
            float f2 = this.z;
            this.h0 = f2;
            int i8 = this.f197a;
            this.i0 = i8;
            int i9 = this.f198b;
            this.j0 = i9;
            float f3 = this.f199c;
            this.k0 = f3;
            if (z2) {
                int i10 = this.p;
                if (i10 != -1) {
                    this.d0 = i10;
                } else {
                    int i11 = this.q;
                    if (i11 != -1) {
                        this.e0 = i11;
                    }
                    i2 = this.r;
                    if (i2 != -1) {
                        this.c0 = i2;
                        z = true;
                    }
                    i3 = this.s;
                    if (i3 != -1) {
                        this.b0 = i3;
                        z = true;
                    }
                    i4 = this.x;
                    if (i4 != -1) {
                        this.g0 = i4;
                    }
                    i5 = this.y;
                    if (i5 != -1) {
                        this.f0 = i5;
                    }
                    if (z) {
                        this.h0 = 1.0f - f2;
                    }
                    if (this.Y && this.R == 1) {
                        if (f3 == -1.0f) {
                            this.k0 = 1.0f - f3;
                            this.i0 = -1;
                            this.j0 = -1;
                        } else if (i8 != -1) {
                            this.j0 = i8;
                            this.i0 = -1;
                            this.k0 = -1.0f;
                        } else if (i9 != -1) {
                            this.i0 = i9;
                            this.j0 = -1;
                            this.k0 = -1.0f;
                        }
                    }
                }
                z = true;
                i2 = this.r;
                if (i2 != -1) {
                }
                i3 = this.s;
                if (i3 != -1) {
                }
                i4 = this.x;
                if (i4 != -1) {
                }
                i5 = this.y;
                if (i5 != -1) {
                }
                if (z) {
                }
                if (this.Y) {
                    if (f3 == -1.0f) {
                    }
                }
            } else {
                int i12 = this.p;
                if (i12 != -1) {
                    this.c0 = i12;
                }
                int i13 = this.q;
                if (i13 != -1) {
                    this.b0 = i13;
                }
                int i14 = this.r;
                if (i14 != -1) {
                    this.d0 = i14;
                }
                int i15 = this.s;
                if (i15 != -1) {
                    this.e0 = i15;
                }
                int i16 = this.x;
                if (i16 != -1) {
                    this.f0 = i16;
                }
                int i17 = this.y;
                if (i17 != -1) {
                    this.g0 = i17;
                }
            }
            if (this.r == -1 && this.s == -1 && this.q == -1 && this.p == -1) {
                int i18 = this.f202f;
                if (i18 != -1) {
                    this.d0 = i18;
                    if (((ViewGroup.MarginLayoutParams) this).rightMargin <= 0 && i7 > 0) {
                        ((ViewGroup.MarginLayoutParams) this).rightMargin = i7;
                    }
                } else {
                    int i19 = this.f203g;
                    if (i19 != -1) {
                        this.e0 = i19;
                        if (((ViewGroup.MarginLayoutParams) this).rightMargin <= 0 && i7 > 0) {
                            ((ViewGroup.MarginLayoutParams) this).rightMargin = i7;
                        }
                    }
                }
                int i20 = this.f200d;
                if (i20 != -1) {
                    this.b0 = i20;
                    if (((ViewGroup.MarginLayoutParams) this).leftMargin > 0 || i6 <= 0) {
                        return;
                    }
                    ((ViewGroup.MarginLayoutParams) this).leftMargin = i6;
                    return;
                }
                int i21 = this.f201e;
                if (i21 != -1) {
                    this.c0 = i21;
                    if (((ViewGroup.MarginLayoutParams) this).leftMargin > 0 || i6 <= 0) {
                        return;
                    }
                    ((ViewGroup.MarginLayoutParams) this).leftMargin = i6;
                }
            }
        }

        public a(int i, int i2) {
            super(i, i2);
            this.f197a = -1;
            this.f198b = -1;
            this.f199c = -1.0f;
            this.f200d = -1;
            this.f201e = -1;
            this.f202f = -1;
            this.f203g = -1;
            this.f204h = -1;
            this.i = -1;
            this.j = -1;
            this.k = -1;
            this.l = -1;
            this.m = -1;
            this.n = 0;
            this.o = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.p = -1;
            this.q = -1;
            this.r = -1;
            this.s = -1;
            this.t = -1;
            this.u = -1;
            this.v = -1;
            this.w = -1;
            this.x = -1;
            this.y = -1;
            this.z = 0.5f;
            this.A = 0.5f;
            this.B = null;
            this.C = 1;
            this.D = -1.0f;
            this.E = -1.0f;
            this.F = 0;
            this.G = 0;
            this.H = 0;
            this.I = 0;
            this.J = 0;
            this.K = 0;
            this.L = 0;
            this.M = 0;
            this.N = 1.0f;
            this.O = 1.0f;
            this.P = -1;
            this.Q = -1;
            this.R = -1;
            this.S = false;
            this.T = false;
            this.U = null;
            this.V = true;
            this.W = true;
            this.X = false;
            this.Y = false;
            this.Z = false;
            this.a0 = false;
            this.b0 = -1;
            this.c0 = -1;
            this.d0 = -1;
            this.e0 = -1;
            this.f0 = -1;
            this.g0 = -1;
            this.h0 = 0.5f;
            this.l0 = new b.h.b.i.d();
        }

        public a(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f197a = -1;
            this.f198b = -1;
            this.f199c = -1.0f;
            this.f200d = -1;
            this.f201e = -1;
            this.f202f = -1;
            this.f203g = -1;
            this.f204h = -1;
            this.i = -1;
            this.j = -1;
            this.k = -1;
            this.l = -1;
            this.m = -1;
            this.n = 0;
            this.o = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.p = -1;
            this.q = -1;
            this.r = -1;
            this.s = -1;
            this.t = -1;
            this.u = -1;
            this.v = -1;
            this.w = -1;
            this.x = -1;
            this.y = -1;
            this.z = 0.5f;
            this.A = 0.5f;
            this.B = null;
            this.C = 1;
            this.D = -1.0f;
            this.E = -1.0f;
            this.F = 0;
            this.G = 0;
            this.H = 0;
            this.I = 0;
            this.J = 0;
            this.K = 0;
            this.L = 0;
            this.M = 0;
            this.N = 1.0f;
            this.O = 1.0f;
            this.P = -1;
            this.Q = -1;
            this.R = -1;
            this.S = false;
            this.T = false;
            this.U = null;
            this.V = true;
            this.W = true;
            this.X = false;
            this.Y = false;
            this.Z = false;
            this.a0 = false;
            this.b0 = -1;
            this.c0 = -1;
            this.d0 = -1;
            this.e0 = -1;
            this.f0 = -1;
            this.g0 = -1;
            this.h0 = 0.5f;
            this.l0 = new b.h.b.i.d();
        }
    }
}