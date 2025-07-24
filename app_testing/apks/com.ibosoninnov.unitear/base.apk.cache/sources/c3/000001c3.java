package b.b.h;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import com.google.android.material.badge.BadgeDrawable;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: LinearLayoutCompat.java */
/* loaded from: classes.dex */
public class i0 extends ViewGroup {
    private static final String ACCESSIBILITY_CLASS_NAME = "androidx.appcompat.widget.LinearLayoutCompat";
    public static final int HORIZONTAL = 0;
    private static final int INDEX_BOTTOM = 2;
    private static final int INDEX_CENTER_VERTICAL = 0;
    private static final int INDEX_FILL = 3;
    private static final int INDEX_TOP = 1;
    public static final int SHOW_DIVIDER_BEGINNING = 1;
    public static final int SHOW_DIVIDER_END = 4;
    public static final int SHOW_DIVIDER_MIDDLE = 2;
    public static final int SHOW_DIVIDER_NONE = 0;
    public static final int VERTICAL = 1;
    private static final int VERTICAL_GRAVITY_COUNT = 4;
    private boolean mBaselineAligned;
    private int mBaselineAlignedChildIndex;
    private int mBaselineChildTop;
    private Drawable mDivider;
    private int mDividerHeight;
    private int mDividerPadding;
    private int mDividerWidth;
    private int mGravity;
    private int[] mMaxAscent;
    private int[] mMaxDescent;
    private int mOrientation;
    private int mShowDividers;
    private int mTotalLength;
    private boolean mUseLargestChild;
    private float mWeightSum;

    public i0(Context context) {
        this(context, null);
    }

    private void forceUniformHeight(int i, int i2) {
        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(getMeasuredHeight(), 1073741824);
        for (int i3 = 0; i3 < i; i3++) {
            View virtualChildAt = getVirtualChildAt(i3);
            if (virtualChildAt.getVisibility() != 8) {
                a aVar = (a) virtualChildAt.getLayoutParams();
                if (((ViewGroup.MarginLayoutParams) aVar).height == -1) {
                    int i4 = ((ViewGroup.MarginLayoutParams) aVar).width;
                    ((ViewGroup.MarginLayoutParams) aVar).width = virtualChildAt.getMeasuredWidth();
                    measureChildWithMargins(virtualChildAt, i2, 0, makeMeasureSpec, 0);
                    ((ViewGroup.MarginLayoutParams) aVar).width = i4;
                }
            }
        }
    }

    private void forceUniformWidth(int i, int i2) {
        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(getMeasuredWidth(), 1073741824);
        for (int i3 = 0; i3 < i; i3++) {
            View virtualChildAt = getVirtualChildAt(i3);
            if (virtualChildAt.getVisibility() != 8) {
                a aVar = (a) virtualChildAt.getLayoutParams();
                if (((ViewGroup.MarginLayoutParams) aVar).width == -1) {
                    int i4 = ((ViewGroup.MarginLayoutParams) aVar).height;
                    ((ViewGroup.MarginLayoutParams) aVar).height = virtualChildAt.getMeasuredHeight();
                    measureChildWithMargins(virtualChildAt, makeMeasureSpec, 0, i2, 0);
                    ((ViewGroup.MarginLayoutParams) aVar).height = i4;
                }
            }
        }
    }

    private void setChildFrame(View view, int i, int i2, int i3, int i4) {
        view.layout(i, i2, i3 + i, i4 + i2);
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return layoutParams instanceof a;
    }

    public void drawDividersHorizontal(Canvas canvas) {
        int right;
        int left;
        int i;
        int left2;
        int virtualChildCount = getVirtualChildCount();
        boolean b2 = e1.b(this);
        for (int i2 = 0; i2 < virtualChildCount; i2++) {
            View virtualChildAt = getVirtualChildAt(i2);
            if (virtualChildAt != null && virtualChildAt.getVisibility() != 8 && hasDividerBeforeChildAt(i2)) {
                a aVar = (a) virtualChildAt.getLayoutParams();
                if (b2) {
                    left2 = virtualChildAt.getRight() + ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
                } else {
                    left2 = (virtualChildAt.getLeft() - ((ViewGroup.MarginLayoutParams) aVar).leftMargin) - this.mDividerWidth;
                }
                drawVerticalDivider(canvas, left2);
            }
        }
        if (hasDividerBeforeChildAt(virtualChildCount)) {
            View virtualChildAt2 = getVirtualChildAt(virtualChildCount - 1);
            if (virtualChildAt2 != null) {
                a aVar2 = (a) virtualChildAt2.getLayoutParams();
                if (b2) {
                    left = virtualChildAt2.getLeft() - ((ViewGroup.MarginLayoutParams) aVar2).leftMargin;
                    i = this.mDividerWidth;
                    right = left - i;
                } else {
                    right = virtualChildAt2.getRight() + ((ViewGroup.MarginLayoutParams) aVar2).rightMargin;
                }
            } else if (b2) {
                right = getPaddingLeft();
            } else {
                left = getWidth() - getPaddingRight();
                i = this.mDividerWidth;
                right = left - i;
            }
            drawVerticalDivider(canvas, right);
        }
    }

    public void drawDividersVertical(Canvas canvas) {
        int bottom;
        int virtualChildCount = getVirtualChildCount();
        for (int i = 0; i < virtualChildCount; i++) {
            View virtualChildAt = getVirtualChildAt(i);
            if (virtualChildAt != null && virtualChildAt.getVisibility() != 8 && hasDividerBeforeChildAt(i)) {
                drawHorizontalDivider(canvas, (virtualChildAt.getTop() - ((ViewGroup.MarginLayoutParams) ((a) virtualChildAt.getLayoutParams())).topMargin) - this.mDividerHeight);
            }
        }
        if (hasDividerBeforeChildAt(virtualChildCount)) {
            View virtualChildAt2 = getVirtualChildAt(virtualChildCount - 1);
            if (virtualChildAt2 == null) {
                bottom = (getHeight() - getPaddingBottom()) - this.mDividerHeight;
            } else {
                bottom = virtualChildAt2.getBottom() + ((ViewGroup.MarginLayoutParams) ((a) virtualChildAt2.getLayoutParams())).bottomMargin;
            }
            drawHorizontalDivider(canvas, bottom);
        }
    }

    public void drawHorizontalDivider(Canvas canvas, int i) {
        this.mDivider.setBounds(getPaddingLeft() + this.mDividerPadding, i, (getWidth() - getPaddingRight()) - this.mDividerPadding, this.mDividerHeight + i);
        this.mDivider.draw(canvas);
    }

    public void drawVerticalDivider(Canvas canvas, int i) {
        this.mDivider.setBounds(i, getPaddingTop() + this.mDividerPadding, this.mDividerWidth + i, (getHeight() - getPaddingBottom()) - this.mDividerPadding);
        this.mDivider.draw(canvas);
    }

    @Override // android.view.View
    public int getBaseline() {
        int i;
        if (this.mBaselineAlignedChildIndex < 0) {
            return super.getBaseline();
        }
        int childCount = getChildCount();
        int i2 = this.mBaselineAlignedChildIndex;
        if (childCount > i2) {
            View childAt = getChildAt(i2);
            int baseline = childAt.getBaseline();
            if (baseline == -1) {
                if (this.mBaselineAlignedChildIndex == 0) {
                    return -1;
                }
                throw new RuntimeException("mBaselineAlignedChildIndex of LinearLayout points to a View that doesn't know how to get its baseline.");
            }
            int i3 = this.mBaselineChildTop;
            if (this.mOrientation == 1 && (i = this.mGravity & 112) != 48) {
                if (i == 16) {
                    i3 += ((((getBottom() - getTop()) - getPaddingTop()) - getPaddingBottom()) - this.mTotalLength) / 2;
                } else if (i == 80) {
                    i3 = ((getBottom() - getTop()) - getPaddingBottom()) - this.mTotalLength;
                }
            }
            return i3 + ((ViewGroup.MarginLayoutParams) ((a) childAt.getLayoutParams())).topMargin + baseline;
        }
        throw new RuntimeException("mBaselineAlignedChildIndex of LinearLayout set to an index that is out of bounds.");
    }

    public int getBaselineAlignedChildIndex() {
        return this.mBaselineAlignedChildIndex;
    }

    public int getChildrenSkipCount(View view, int i) {
        return 0;
    }

    public Drawable getDividerDrawable() {
        return this.mDivider;
    }

    public int getDividerPadding() {
        return this.mDividerPadding;
    }

    public int getDividerWidth() {
        return this.mDividerWidth;
    }

    public int getGravity() {
        return this.mGravity;
    }

    public int getLocationOffset(View view) {
        return 0;
    }

    public int getNextLocationOffset(View view) {
        return 0;
    }

    public int getOrientation() {
        return this.mOrientation;
    }

    public int getShowDividers() {
        return this.mShowDividers;
    }

    public View getVirtualChildAt(int i) {
        return getChildAt(i);
    }

    public int getVirtualChildCount() {
        return getChildCount();
    }

    public float getWeightSum() {
        return this.mWeightSum;
    }

    public boolean hasDividerBeforeChildAt(int i) {
        if (i == 0) {
            return (this.mShowDividers & 1) != 0;
        } else if (i == getChildCount()) {
            return (this.mShowDividers & 4) != 0;
        } else if ((this.mShowDividers & 2) != 0) {
            for (int i2 = i - 1; i2 >= 0; i2--) {
                if (getChildAt(i2).getVisibility() != 8) {
                    return true;
                }
            }
            return false;
        } else {
            return false;
        }
    }

    public boolean isBaselineAligned() {
        return this.mBaselineAligned;
    }

    public boolean isMeasureWithLargestChildEnabled() {
        return this.mUseLargestChild;
    }

    /* JADX WARN: Removed duplicated region for block: B:29:0x00b2  */
    /* JADX WARN: Removed duplicated region for block: B:32:0x00bb  */
    /* JADX WARN: Removed duplicated region for block: B:44:0x00f0  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x0104  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void layoutHorizontal(int i, int i2, int i3, int i4) {
        int paddingLeft;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        boolean z;
        int i10;
        int i11;
        int i12;
        int i13;
        boolean b2 = e1.b(this);
        int paddingTop = getPaddingTop();
        int i14 = i4 - i2;
        int paddingBottom = i14 - getPaddingBottom();
        int paddingBottom2 = (i14 - paddingTop) - getPaddingBottom();
        int virtualChildCount = getVirtualChildCount();
        int i15 = this.mGravity;
        int i16 = 8388615 & i15;
        int i17 = i15 & 112;
        boolean z2 = this.mBaselineAligned;
        int[] iArr = this.mMaxAscent;
        int[] iArr2 = this.mMaxDescent;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        int absoluteGravity = Gravity.getAbsoluteGravity(i16, getLayoutDirection());
        boolean z3 = true;
        if (absoluteGravity == 1) {
            paddingLeft = getPaddingLeft() + (((i3 - i) - this.mTotalLength) / 2);
        } else if (absoluteGravity != 5) {
            paddingLeft = getPaddingLeft();
        } else {
            paddingLeft = ((getPaddingLeft() + i3) - i) - this.mTotalLength;
        }
        if (b2) {
            i5 = virtualChildCount - 1;
            i6 = -1;
        } else {
            i5 = 0;
            i6 = 1;
        }
        int i18 = 0;
        while (i18 < virtualChildCount) {
            int i19 = (i6 * i18) + i5;
            View virtualChildAt = getVirtualChildAt(i19);
            if (virtualChildAt == null) {
                paddingLeft = measureNullChild(i19) + paddingLeft;
                z = z3;
                i7 = paddingTop;
                i8 = virtualChildCount;
                i9 = i17;
            } else if (virtualChildAt.getVisibility() != 8) {
                int measuredWidth = virtualChildAt.getMeasuredWidth();
                int measuredHeight = virtualChildAt.getMeasuredHeight();
                a aVar = (a) virtualChildAt.getLayoutParams();
                int i20 = i18;
                if (z2) {
                    i8 = virtualChildCount;
                    if (((ViewGroup.MarginLayoutParams) aVar).height != -1) {
                        i10 = virtualChildAt.getBaseline();
                        i11 = aVar.f861b;
                        if (i11 < 0) {
                            i11 = i17;
                        }
                        i12 = i11 & 112;
                        i9 = i17;
                        if (i12 == 16) {
                            if (i12 == 48) {
                                i13 = ((ViewGroup.MarginLayoutParams) aVar).topMargin + paddingTop;
                                if (i10 != -1) {
                                    z = true;
                                    i13 = (iArr[1] - i10) + i13;
                                }
                            } else if (i12 != 80) {
                                i13 = paddingTop;
                            } else {
                                i13 = (paddingBottom - measuredHeight) - ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
                                if (i10 != -1) {
                                    i13 -= iArr2[2] - (virtualChildAt.getMeasuredHeight() - i10);
                                }
                            }
                            z = true;
                        } else {
                            z = true;
                            i13 = ((((paddingBottom2 - measuredHeight) / 2) + paddingTop) + ((ViewGroup.MarginLayoutParams) aVar).topMargin) - ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
                        }
                        if (hasDividerBeforeChildAt(i19)) {
                            paddingLeft += this.mDividerWidth;
                        }
                        int i21 = ((ViewGroup.MarginLayoutParams) aVar).leftMargin + paddingLeft;
                        i7 = paddingTop;
                        setChildFrame(virtualChildAt, getLocationOffset(virtualChildAt) + i21, i13, measuredWidth, measuredHeight);
                        paddingLeft = getNextLocationOffset(virtualChildAt) + measuredWidth + ((ViewGroup.MarginLayoutParams) aVar).rightMargin + i21;
                        i18 = getChildrenSkipCount(virtualChildAt, i19) + i20;
                        i18++;
                        virtualChildCount = i8;
                        i17 = i9;
                        z3 = z;
                        paddingTop = i7;
                    }
                } else {
                    i8 = virtualChildCount;
                }
                i10 = -1;
                i11 = aVar.f861b;
                if (i11 < 0) {
                }
                i12 = i11 & 112;
                i9 = i17;
                if (i12 == 16) {
                }
                if (hasDividerBeforeChildAt(i19)) {
                }
                int i212 = ((ViewGroup.MarginLayoutParams) aVar).leftMargin + paddingLeft;
                i7 = paddingTop;
                setChildFrame(virtualChildAt, getLocationOffset(virtualChildAt) + i212, i13, measuredWidth, measuredHeight);
                paddingLeft = getNextLocationOffset(virtualChildAt) + measuredWidth + ((ViewGroup.MarginLayoutParams) aVar).rightMargin + i212;
                i18 = getChildrenSkipCount(virtualChildAt, i19) + i20;
                i18++;
                virtualChildCount = i8;
                i17 = i9;
                z3 = z;
                paddingTop = i7;
            } else {
                i7 = paddingTop;
                i8 = virtualChildCount;
                i9 = i17;
                z = true;
            }
            i18++;
            virtualChildCount = i8;
            i17 = i9;
            z3 = z;
            paddingTop = i7;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:29:0x00a2  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void layoutVertical(int i, int i2, int i3, int i4) {
        int paddingTop;
        int i5;
        int i6;
        int i7;
        int i8;
        int paddingLeft = getPaddingLeft();
        int i9 = i3 - i;
        int paddingRight = i9 - getPaddingRight();
        int paddingRight2 = (i9 - paddingLeft) - getPaddingRight();
        int virtualChildCount = getVirtualChildCount();
        int i10 = this.mGravity;
        int i11 = i10 & 112;
        int i12 = i10 & 8388615;
        if (i11 == 16) {
            paddingTop = getPaddingTop() + (((i4 - i2) - this.mTotalLength) / 2);
        } else if (i11 != 80) {
            paddingTop = getPaddingTop();
        } else {
            paddingTop = ((getPaddingTop() + i4) - i2) - this.mTotalLength;
        }
        int i13 = 0;
        while (i13 < virtualChildCount) {
            View virtualChildAt = getVirtualChildAt(i13);
            if (virtualChildAt == null) {
                paddingTop = measureNullChild(i13) + paddingTop;
            } else if (virtualChildAt.getVisibility() != 8) {
                int measuredWidth = virtualChildAt.getMeasuredWidth();
                int measuredHeight = virtualChildAt.getMeasuredHeight();
                a aVar = (a) virtualChildAt.getLayoutParams();
                int i14 = aVar.f861b;
                if (i14 < 0) {
                    i14 = i12;
                }
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                int absoluteGravity = Gravity.getAbsoluteGravity(i14, getLayoutDirection()) & 7;
                if (absoluteGravity == 1) {
                    i5 = ((paddingRight2 - measuredWidth) / 2) + paddingLeft + ((ViewGroup.MarginLayoutParams) aVar).leftMargin;
                    i6 = ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
                } else if (absoluteGravity != 5) {
                    i7 = ((ViewGroup.MarginLayoutParams) aVar).leftMargin + paddingLeft;
                    int i15 = i7;
                    if (hasDividerBeforeChildAt(i13)) {
                        paddingTop += this.mDividerHeight;
                    }
                    int i16 = paddingTop + ((ViewGroup.MarginLayoutParams) aVar).topMargin;
                    setChildFrame(virtualChildAt, i15, getLocationOffset(virtualChildAt) + i16, measuredWidth, measuredHeight);
                    paddingTop = getNextLocationOffset(virtualChildAt) + measuredHeight + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin + i16;
                    i13 += getChildrenSkipCount(virtualChildAt, i13);
                    i8 = 1;
                    i13 += i8;
                } else {
                    i5 = paddingRight - measuredWidth;
                    i6 = ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
                }
                i7 = i5 - i6;
                int i152 = i7;
                if (hasDividerBeforeChildAt(i13)) {
                }
                int i162 = paddingTop + ((ViewGroup.MarginLayoutParams) aVar).topMargin;
                setChildFrame(virtualChildAt, i152, getLocationOffset(virtualChildAt) + i162, measuredWidth, measuredHeight);
                paddingTop = getNextLocationOffset(virtualChildAt) + measuredHeight + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin + i162;
                i13 += getChildrenSkipCount(virtualChildAt, i13);
                i8 = 1;
                i13 += i8;
            }
            i8 = 1;
            i13 += i8;
        }
    }

    public void measureChildBeforeLayout(View view, int i, int i2, int i3, int i4, int i5) {
        measureChildWithMargins(view, i2, i3, i4, i5);
    }

    /* JADX WARN: Removed duplicated region for block: B:197:0x0454  */
    /* JADX WARN: Removed duplicated region for block: B:67:0x0199  */
    /* JADX WARN: Removed duplicated region for block: B:77:0x01cd  */
    /* JADX WARN: Removed duplicated region for block: B:81:0x01d8  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void measureHorizontal(int i, int i2) {
        int[] iArr;
        int i3;
        int max;
        int i4;
        int i5;
        int max2;
        int i6;
        int i7;
        int i8;
        float f2;
        int i9;
        boolean z;
        int baseline;
        int i10;
        int i11;
        int i12;
        char c2;
        int i13;
        int i14;
        boolean z2;
        boolean z3;
        View view;
        int i15;
        boolean z4;
        int measuredHeight;
        int childrenSkipCount;
        int baseline2;
        int i16;
        this.mTotalLength = 0;
        int virtualChildCount = getVirtualChildCount();
        int mode = View.MeasureSpec.getMode(i);
        int mode2 = View.MeasureSpec.getMode(i2);
        if (this.mMaxAscent == null || this.mMaxDescent == null) {
            this.mMaxAscent = new int[4];
            this.mMaxDescent = new int[4];
        }
        int[] iArr2 = this.mMaxAscent;
        int[] iArr3 = this.mMaxDescent;
        iArr2[3] = -1;
        iArr2[2] = -1;
        iArr2[1] = -1;
        iArr2[0] = -1;
        iArr3[3] = -1;
        iArr3[2] = -1;
        iArr3[1] = -1;
        iArr3[0] = -1;
        boolean z5 = this.mBaselineAligned;
        boolean z6 = this.mUseLargestChild;
        int i17 = 1073741824;
        boolean z7 = mode == 1073741824;
        int i18 = 0;
        int i19 = 0;
        int i20 = 0;
        int i21 = 0;
        int i22 = 0;
        boolean z8 = false;
        int i23 = 0;
        boolean z9 = false;
        boolean z10 = true;
        float f3 = 0.0f;
        while (true) {
            iArr = iArr3;
            if (i18 >= virtualChildCount) {
                break;
            }
            View virtualChildAt = getVirtualChildAt(i18);
            if (virtualChildAt == null) {
                this.mTotalLength = measureNullChild(i18) + this.mTotalLength;
            } else if (virtualChildAt.getVisibility() == 8) {
                i18 += getChildrenSkipCount(virtualChildAt, i18);
            } else {
                if (hasDividerBeforeChildAt(i18)) {
                    this.mTotalLength += this.mDividerWidth;
                }
                a aVar = (a) virtualChildAt.getLayoutParams();
                float f4 = aVar.f860a;
                float f5 = f3 + f4;
                if (mode == i17 && ((ViewGroup.MarginLayoutParams) aVar).width == 0 && f4 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    if (z7) {
                        this.mTotalLength = ((ViewGroup.MarginLayoutParams) aVar).leftMargin + ((ViewGroup.MarginLayoutParams) aVar).rightMargin + this.mTotalLength;
                    } else {
                        int i24 = this.mTotalLength;
                        this.mTotalLength = Math.max(i24, ((ViewGroup.MarginLayoutParams) aVar).leftMargin + i24 + ((ViewGroup.MarginLayoutParams) aVar).rightMargin);
                    }
                    if (z5) {
                        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, 0);
                        virtualChildAt.measure(makeMeasureSpec, makeMeasureSpec);
                        i14 = i18;
                        z2 = z6;
                        z3 = z5;
                        view = virtualChildAt;
                    } else {
                        i14 = i18;
                        z2 = z6;
                        z3 = z5;
                        view = virtualChildAt;
                        z8 = true;
                        i15 = 1073741824;
                        if (mode2 == i15 && ((ViewGroup.MarginLayoutParams) aVar).height == -1) {
                            z4 = true;
                            z9 = true;
                        } else {
                            z4 = false;
                        }
                        int i25 = ((ViewGroup.MarginLayoutParams) aVar).topMargin + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
                        measuredHeight = view.getMeasuredHeight() + i25;
                        i23 = View.combineMeasuredStates(i23, view.getMeasuredState());
                        if (z3 && (baseline2 = view.getBaseline()) != -1) {
                            i16 = aVar.f861b;
                            if (i16 < 0) {
                                i16 = this.mGravity;
                            }
                            int i26 = (((i16 & 112) >> 4) & (-2)) >> 1;
                            iArr2[i26] = Math.max(iArr2[i26], baseline2);
                            iArr[i26] = Math.max(iArr[i26], measuredHeight - baseline2);
                        }
                        i20 = Math.max(i20, measuredHeight);
                        z10 = !z10 && ((ViewGroup.MarginLayoutParams) aVar).height == -1;
                        if (aVar.f860a <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            if (!z4) {
                                i25 = measuredHeight;
                            }
                            i22 = Math.max(i22, i25);
                        } else {
                            int i27 = i22;
                            if (!z4) {
                                i25 = measuredHeight;
                            }
                            i21 = Math.max(i21, i25);
                            i22 = i27;
                        }
                        int i28 = i14;
                        childrenSkipCount = getChildrenSkipCount(view, i28) + i28;
                        f3 = f5;
                        int i29 = childrenSkipCount + 1;
                        iArr3 = iArr;
                        z6 = z2;
                        z5 = z3;
                        i17 = i15;
                        i18 = i29;
                    }
                } else {
                    if (((ViewGroup.MarginLayoutParams) aVar).width != 0 || f4 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        c2 = 65534;
                        i13 = Integer.MIN_VALUE;
                    } else {
                        c2 = 65534;
                        ((ViewGroup.MarginLayoutParams) aVar).width = -2;
                        i13 = 0;
                    }
                    i14 = i18;
                    int i30 = i13;
                    z2 = z6;
                    z3 = z5;
                    measureChildBeforeLayout(virtualChildAt, i14, i, f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? this.mTotalLength : 0, i2, 0);
                    if (i30 != Integer.MIN_VALUE) {
                        ((ViewGroup.MarginLayoutParams) aVar).width = i30;
                    }
                    int measuredWidth = virtualChildAt.getMeasuredWidth();
                    if (z7) {
                        view = virtualChildAt;
                        this.mTotalLength = getNextLocationOffset(view) + ((ViewGroup.MarginLayoutParams) aVar).leftMargin + measuredWidth + ((ViewGroup.MarginLayoutParams) aVar).rightMargin + this.mTotalLength;
                    } else {
                        view = virtualChildAt;
                        int i31 = this.mTotalLength;
                        this.mTotalLength = Math.max(i31, getNextLocationOffset(view) + i31 + measuredWidth + ((ViewGroup.MarginLayoutParams) aVar).leftMargin + ((ViewGroup.MarginLayoutParams) aVar).rightMargin);
                    }
                    if (z2) {
                        i19 = Math.max(measuredWidth, i19);
                    }
                }
                i15 = 1073741824;
                if (mode2 == i15) {
                }
                z4 = false;
                int i252 = ((ViewGroup.MarginLayoutParams) aVar).topMargin + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin;
                measuredHeight = view.getMeasuredHeight() + i252;
                i23 = View.combineMeasuredStates(i23, view.getMeasuredState());
                if (z3) {
                    i16 = aVar.f861b;
                    if (i16 < 0) {
                    }
                    int i262 = (((i16 & 112) >> 4) & (-2)) >> 1;
                    iArr2[i262] = Math.max(iArr2[i262], baseline2);
                    iArr[i262] = Math.max(iArr[i262], measuredHeight - baseline2);
                }
                i20 = Math.max(i20, measuredHeight);
                if (z10) {
                }
                if (aVar.f860a <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                }
                int i282 = i14;
                childrenSkipCount = getChildrenSkipCount(view, i282) + i282;
                f3 = f5;
                int i292 = childrenSkipCount + 1;
                iArr3 = iArr;
                z6 = z2;
                z5 = z3;
                i17 = i15;
                i18 = i292;
            }
            z2 = z6;
            z3 = z5;
            int i32 = i17;
            childrenSkipCount = i18;
            i15 = i32;
            int i2922 = childrenSkipCount + 1;
            iArr3 = iArr;
            z6 = z2;
            z5 = z3;
            i17 = i15;
            i18 = i2922;
        }
        boolean z11 = z6;
        boolean z12 = z5;
        int i33 = i20;
        int i34 = i21;
        int i35 = i22;
        int i36 = i23;
        if (this.mTotalLength > 0 && hasDividerBeforeChildAt(virtualChildCount)) {
            this.mTotalLength += this.mDividerWidth;
        }
        if (iArr2[1] == -1 && iArr2[0] == -1 && iArr2[2] == -1 && iArr2[3] == -1) {
            max = i33;
            i3 = i36;
        } else {
            i3 = i36;
            max = Math.max(i33, Math.max(iArr[3], Math.max(iArr[0], Math.max(iArr[1], iArr[2]))) + Math.max(iArr2[3], Math.max(iArr2[0], Math.max(iArr2[1], iArr2[2]))));
        }
        if (z11 && (mode == Integer.MIN_VALUE || mode == 0)) {
            this.mTotalLength = 0;
            int i37 = 0;
            while (i37 < virtualChildCount) {
                View virtualChildAt2 = getVirtualChildAt(i37);
                if (virtualChildAt2 == null) {
                    this.mTotalLength = measureNullChild(i37) + this.mTotalLength;
                } else if (virtualChildAt2.getVisibility() == 8) {
                    i37 += getChildrenSkipCount(virtualChildAt2, i37);
                } else {
                    a aVar2 = (a) virtualChildAt2.getLayoutParams();
                    if (z7) {
                        this.mTotalLength = getNextLocationOffset(virtualChildAt2) + ((ViewGroup.MarginLayoutParams) aVar2).leftMargin + i19 + ((ViewGroup.MarginLayoutParams) aVar2).rightMargin + this.mTotalLength;
                    } else {
                        int i38 = this.mTotalLength;
                        i12 = max;
                        this.mTotalLength = Math.max(i38, getNextLocationOffset(virtualChildAt2) + i38 + i19 + ((ViewGroup.MarginLayoutParams) aVar2).leftMargin + ((ViewGroup.MarginLayoutParams) aVar2).rightMargin);
                        i37++;
                        max = i12;
                    }
                }
                i12 = max;
                i37++;
                max = i12;
            }
        }
        int i39 = max;
        int paddingRight = getPaddingRight() + getPaddingLeft() + this.mTotalLength;
        this.mTotalLength = paddingRight;
        int resolveSizeAndState = View.resolveSizeAndState(Math.max(paddingRight, getSuggestedMinimumWidth()), i, 0);
        int i40 = (16777215 & resolveSizeAndState) - this.mTotalLength;
        if (!z8 && (i40 == 0 || f3 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            i6 = Math.max(i34, i35);
            if (z11 && mode != 1073741824) {
                for (int i41 = 0; i41 < virtualChildCount; i41++) {
                    View virtualChildAt3 = getVirtualChildAt(i41);
                    if (virtualChildAt3 != null && virtualChildAt3.getVisibility() != 8 && ((a) virtualChildAt3.getLayoutParams()).f860a > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        virtualChildAt3.measure(View.MeasureSpec.makeMeasureSpec(i19, 1073741824), View.MeasureSpec.makeMeasureSpec(virtualChildAt3.getMeasuredHeight(), 1073741824));
                    }
                }
            }
            i4 = i2;
            i5 = virtualChildCount;
            max2 = i39;
        } else {
            float f6 = this.mWeightSum;
            if (f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                f3 = f6;
            }
            iArr2[3] = -1;
            iArr2[2] = -1;
            iArr2[1] = -1;
            iArr2[0] = -1;
            iArr[3] = -1;
            iArr[2] = -1;
            iArr[1] = -1;
            iArr[0] = -1;
            this.mTotalLength = 0;
            int i42 = i34;
            int i43 = -1;
            int i44 = i3;
            int i45 = 0;
            while (i45 < virtualChildCount) {
                View virtualChildAt4 = getVirtualChildAt(i45);
                if (virtualChildAt4 == null || virtualChildAt4.getVisibility() == 8) {
                    i7 = i40;
                    i8 = virtualChildCount;
                } else {
                    a aVar3 = (a) virtualChildAt4.getLayoutParams();
                    float f7 = aVar3.f860a;
                    if (f7 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        int i46 = (int) ((i40 * f7) / f3);
                        float f8 = f3 - f7;
                        int i47 = i40 - i46;
                        i8 = virtualChildCount;
                        int childMeasureSpec = ViewGroup.getChildMeasureSpec(i2, getPaddingBottom() + getPaddingTop() + ((ViewGroup.MarginLayoutParams) aVar3).topMargin + ((ViewGroup.MarginLayoutParams) aVar3).bottomMargin, ((ViewGroup.MarginLayoutParams) aVar3).height);
                        if (((ViewGroup.MarginLayoutParams) aVar3).width == 0) {
                            i11 = 1073741824;
                            if (mode == 1073741824) {
                                if (i46 <= 0) {
                                    i46 = 0;
                                }
                                virtualChildAt4.measure(View.MeasureSpec.makeMeasureSpec(i46, 1073741824), childMeasureSpec);
                                i44 = View.combineMeasuredStates(i44, virtualChildAt4.getMeasuredState() & (-16777216));
                                f3 = f8;
                                i7 = i47;
                            }
                        } else {
                            i11 = 1073741824;
                        }
                        int measuredWidth2 = virtualChildAt4.getMeasuredWidth() + i46;
                        if (measuredWidth2 < 0) {
                            measuredWidth2 = 0;
                        }
                        virtualChildAt4.measure(View.MeasureSpec.makeMeasureSpec(measuredWidth2, i11), childMeasureSpec);
                        i44 = View.combineMeasuredStates(i44, virtualChildAt4.getMeasuredState() & (-16777216));
                        f3 = f8;
                        i7 = i47;
                    } else {
                        i7 = i40;
                        i8 = virtualChildCount;
                    }
                    if (z7) {
                        this.mTotalLength = getNextLocationOffset(virtualChildAt4) + virtualChildAt4.getMeasuredWidth() + ((ViewGroup.MarginLayoutParams) aVar3).leftMargin + ((ViewGroup.MarginLayoutParams) aVar3).rightMargin + this.mTotalLength;
                        f2 = f3;
                    } else {
                        int i48 = this.mTotalLength;
                        f2 = f3;
                        this.mTotalLength = Math.max(i48, getNextLocationOffset(virtualChildAt4) + virtualChildAt4.getMeasuredWidth() + i48 + ((ViewGroup.MarginLayoutParams) aVar3).leftMargin + ((ViewGroup.MarginLayoutParams) aVar3).rightMargin);
                    }
                    boolean z13 = mode2 != 1073741824 && ((ViewGroup.MarginLayoutParams) aVar3).height == -1;
                    int i49 = ((ViewGroup.MarginLayoutParams) aVar3).topMargin + ((ViewGroup.MarginLayoutParams) aVar3).bottomMargin;
                    int measuredHeight2 = virtualChildAt4.getMeasuredHeight() + i49;
                    i43 = Math.max(i43, measuredHeight2);
                    if (!z13) {
                        i49 = measuredHeight2;
                    }
                    int max3 = Math.max(i42, i49);
                    if (z10) {
                        i9 = -1;
                        if (((ViewGroup.MarginLayoutParams) aVar3).height == -1) {
                            z = true;
                            if (z12 && (baseline = virtualChildAt4.getBaseline()) != i9) {
                                i10 = aVar3.f861b;
                                if (i10 < 0) {
                                    i10 = this.mGravity;
                                }
                                int i50 = (((i10 & 112) >> 4) & (-2)) >> 1;
                                iArr2[i50] = Math.max(iArr2[i50], baseline);
                                iArr[i50] = Math.max(iArr[i50], measuredHeight2 - baseline);
                            }
                            z10 = z;
                            i42 = max3;
                            f3 = f2;
                        }
                    } else {
                        i9 = -1;
                    }
                    z = false;
                    if (z12) {
                        i10 = aVar3.f861b;
                        if (i10 < 0) {
                        }
                        int i502 = (((i10 & 112) >> 4) & (-2)) >> 1;
                        iArr2[i502] = Math.max(iArr2[i502], baseline);
                        iArr[i502] = Math.max(iArr[i502], measuredHeight2 - baseline);
                    }
                    z10 = z;
                    i42 = max3;
                    f3 = f2;
                }
                i45++;
                i40 = i7;
                virtualChildCount = i8;
            }
            i4 = i2;
            i5 = virtualChildCount;
            this.mTotalLength = getPaddingRight() + getPaddingLeft() + this.mTotalLength;
            max2 = (iArr2[1] == -1 && iArr2[0] == -1 && iArr2[2] == -1 && iArr2[3] == -1) ? i43 : Math.max(i43, Math.max(iArr[3], Math.max(iArr[0], Math.max(iArr[1], iArr[2]))) + Math.max(iArr2[3], Math.max(iArr2[0], Math.max(iArr2[1], iArr2[2]))));
            i6 = i42;
            i3 = i44;
        }
        if (z10 || mode2 == 1073741824) {
            i6 = max2;
        }
        setMeasuredDimension(resolveSizeAndState | (i3 & (-16777216)), View.resolveSizeAndState(Math.max(getPaddingBottom() + getPaddingTop() + i6, getSuggestedMinimumHeight()), i4, i3 << 16));
        if (z9) {
            forceUniformHeight(i5, i);
        }
    }

    public int measureNullChild(int i) {
        return 0;
    }

    /* JADX WARN: Removed duplicated region for block: B:150:0x0327  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void measureVertical(int i, int i2) {
        int i3;
        int i4;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        boolean z;
        int i10;
        int i11;
        int i12;
        int i13;
        int i14;
        int i15;
        int i16;
        int i17;
        int i18;
        View view;
        int max;
        boolean z2;
        int max2;
        this.mTotalLength = 0;
        int virtualChildCount = getVirtualChildCount();
        int mode = View.MeasureSpec.getMode(i);
        int mode2 = View.MeasureSpec.getMode(i2);
        int i19 = this.mBaselineAlignedChildIndex;
        boolean z3 = this.mUseLargestChild;
        int i20 = 0;
        int i21 = 0;
        int i22 = 0;
        int i23 = 0;
        int i24 = 0;
        int i25 = 0;
        boolean z4 = false;
        boolean z5 = false;
        float f2 = 0.0f;
        boolean z6 = true;
        while (true) {
            int i26 = 8;
            int i27 = i23;
            if (i25 < virtualChildCount) {
                View virtualChildAt = getVirtualChildAt(i25);
                if (virtualChildAt == null) {
                    this.mTotalLength = measureNullChild(i25) + this.mTotalLength;
                    i14 = virtualChildCount;
                    i15 = mode2;
                    i23 = i27;
                } else {
                    int i28 = i20;
                    if (virtualChildAt.getVisibility() == 8) {
                        i25 += getChildrenSkipCount(virtualChildAt, i25);
                        i14 = virtualChildCount;
                        i23 = i27;
                        i20 = i28;
                        i15 = mode2;
                    } else {
                        if (hasDividerBeforeChildAt(i25)) {
                            this.mTotalLength += this.mDividerHeight;
                        }
                        a aVar = (a) virtualChildAt.getLayoutParams();
                        float f3 = aVar.f860a;
                        float f4 = f2 + f3;
                        if (mode2 == 1073741824 && ((ViewGroup.MarginLayoutParams) aVar).height == 0 && f3 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            int i29 = this.mTotalLength;
                            this.mTotalLength = Math.max(i29, ((ViewGroup.MarginLayoutParams) aVar).topMargin + i29 + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin);
                            max = i22;
                            view = virtualChildAt;
                            i17 = i24;
                            z4 = true;
                            i12 = i28;
                            i13 = i21;
                            i14 = virtualChildCount;
                            i15 = mode2;
                            i16 = i27;
                            i18 = i25;
                        } else {
                            int i30 = i21;
                            if (((ViewGroup.MarginLayoutParams) aVar).height != 0 || f3 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                i11 = Integer.MIN_VALUE;
                            } else {
                                ((ViewGroup.MarginLayoutParams) aVar).height = -2;
                                i11 = 0;
                            }
                            i12 = i28;
                            int i31 = i11;
                            i13 = i30;
                            int i32 = i22;
                            i14 = virtualChildCount;
                            i15 = mode2;
                            i16 = i27;
                            i17 = i24;
                            i18 = i25;
                            measureChildBeforeLayout(virtualChildAt, i25, i, 0, i2, f4 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? this.mTotalLength : 0);
                            if (i31 != Integer.MIN_VALUE) {
                                ((ViewGroup.MarginLayoutParams) aVar).height = i31;
                            }
                            int measuredHeight = virtualChildAt.getMeasuredHeight();
                            int i33 = this.mTotalLength;
                            view = virtualChildAt;
                            this.mTotalLength = Math.max(i33, getNextLocationOffset(view) + i33 + measuredHeight + ((ViewGroup.MarginLayoutParams) aVar).topMargin + ((ViewGroup.MarginLayoutParams) aVar).bottomMargin);
                            max = z3 ? Math.max(measuredHeight, i32) : i32;
                        }
                        if (i19 >= 0 && i19 == i18 + 1) {
                            this.mBaselineChildTop = this.mTotalLength;
                        }
                        if (i18 < i19 && aVar.f860a > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            throw new RuntimeException("A child of LinearLayout with index less than mBaselineAlignedChildIndex has weight > 0, which won't work.  Either remove the weight, or don't set mBaselineAlignedChildIndex.");
                        }
                        if (mode == 1073741824 || ((ViewGroup.MarginLayoutParams) aVar).width != -1) {
                            z2 = false;
                        } else {
                            z2 = true;
                            z5 = true;
                        }
                        int i34 = ((ViewGroup.MarginLayoutParams) aVar).leftMargin + ((ViewGroup.MarginLayoutParams) aVar).rightMargin;
                        int measuredWidth = view.getMeasuredWidth() + i34;
                        int max3 = Math.max(i13, measuredWidth);
                        int combineMeasuredStates = View.combineMeasuredStates(i12, view.getMeasuredState());
                        z6 = z6 && ((ViewGroup.MarginLayoutParams) aVar).width == -1;
                        if (aVar.f860a > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            if (!z2) {
                                i34 = measuredWidth;
                            }
                            i23 = Math.max(i16, i34);
                            max2 = i17;
                        } else {
                            if (!z2) {
                                i34 = measuredWidth;
                            }
                            max2 = Math.max(i17, i34);
                            i23 = i16;
                        }
                        i22 = max;
                        f2 = f4;
                        i24 = max2;
                        i20 = combineMeasuredStates;
                        i25 = getChildrenSkipCount(view, i18) + i18;
                        i21 = max3;
                    }
                }
                i25++;
                mode2 = i15;
                virtualChildCount = i14;
            } else {
                int i35 = i20;
                int i36 = i22;
                int i37 = i24;
                int i38 = virtualChildCount;
                int i39 = mode2;
                int i40 = i21;
                if (this.mTotalLength > 0) {
                    i3 = i38;
                    if (hasDividerBeforeChildAt(i3)) {
                        this.mTotalLength += this.mDividerHeight;
                    }
                } else {
                    i3 = i38;
                }
                if (z3 && (i39 == Integer.MIN_VALUE || i39 == 0)) {
                    this.mTotalLength = 0;
                    int i41 = 0;
                    while (i41 < i3) {
                        View virtualChildAt2 = getVirtualChildAt(i41);
                        if (virtualChildAt2 == null) {
                            this.mTotalLength = measureNullChild(i41) + this.mTotalLength;
                        } else if (virtualChildAt2.getVisibility() == i26) {
                            i41 += getChildrenSkipCount(virtualChildAt2, i41);
                        } else {
                            a aVar2 = (a) virtualChildAt2.getLayoutParams();
                            int i42 = this.mTotalLength;
                            this.mTotalLength = Math.max(i42, getNextLocationOffset(virtualChildAt2) + i42 + i36 + ((ViewGroup.MarginLayoutParams) aVar2).topMargin + ((ViewGroup.MarginLayoutParams) aVar2).bottomMargin);
                        }
                        i41++;
                        i26 = 8;
                    }
                }
                int paddingBottom = getPaddingBottom() + getPaddingTop() + this.mTotalLength;
                this.mTotalLength = paddingBottom;
                int resolveSizeAndState = View.resolveSizeAndState(Math.max(paddingBottom, getSuggestedMinimumHeight()), i2, 0);
                int i43 = (16777215 & resolveSizeAndState) - this.mTotalLength;
                if (!z4 && (i43 == 0 || f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
                    i6 = Math.max(i37, i27);
                    if (z3 && i39 != 1073741824) {
                        for (int i44 = 0; i44 < i3; i44++) {
                            View virtualChildAt3 = getVirtualChildAt(i44);
                            if (virtualChildAt3 != null && virtualChildAt3.getVisibility() != 8 && ((a) virtualChildAt3.getLayoutParams()).f860a > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                virtualChildAt3.measure(View.MeasureSpec.makeMeasureSpec(virtualChildAt3.getMeasuredWidth(), 1073741824), View.MeasureSpec.makeMeasureSpec(i36, 1073741824));
                            }
                        }
                    }
                    i5 = i;
                    i4 = i35;
                } else {
                    float f5 = this.mWeightSum;
                    if (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        f2 = f5;
                    }
                    this.mTotalLength = 0;
                    int i45 = i43;
                    int i46 = i37;
                    i4 = i35;
                    int i47 = 0;
                    while (i47 < i3) {
                        View virtualChildAt4 = getVirtualChildAt(i47);
                        if (virtualChildAt4.getVisibility() == 8) {
                            i7 = i45;
                        } else {
                            a aVar3 = (a) virtualChildAt4.getLayoutParams();
                            float f6 = aVar3.f860a;
                            if (f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                int i48 = (int) ((i45 * f6) / f2);
                                float f7 = f2 - f6;
                                i7 = i45 - i48;
                                int childMeasureSpec = ViewGroup.getChildMeasureSpec(i, getPaddingRight() + getPaddingLeft() + ((ViewGroup.MarginLayoutParams) aVar3).leftMargin + ((ViewGroup.MarginLayoutParams) aVar3).rightMargin, ((ViewGroup.MarginLayoutParams) aVar3).width);
                                if (((ViewGroup.MarginLayoutParams) aVar3).height == 0) {
                                    i10 = 1073741824;
                                    if (i39 == 1073741824) {
                                        if (i48 <= 0) {
                                            i48 = 0;
                                        }
                                        virtualChildAt4.measure(childMeasureSpec, View.MeasureSpec.makeMeasureSpec(i48, 1073741824));
                                        i4 = View.combineMeasuredStates(i4, virtualChildAt4.getMeasuredState() & (-256));
                                        f2 = f7;
                                    }
                                } else {
                                    i10 = 1073741824;
                                }
                                int measuredHeight2 = virtualChildAt4.getMeasuredHeight() + i48;
                                if (measuredHeight2 < 0) {
                                    measuredHeight2 = 0;
                                }
                                virtualChildAt4.measure(childMeasureSpec, View.MeasureSpec.makeMeasureSpec(measuredHeight2, i10));
                                i4 = View.combineMeasuredStates(i4, virtualChildAt4.getMeasuredState() & (-256));
                                f2 = f7;
                            } else {
                                i7 = i45;
                            }
                            int i49 = ((ViewGroup.MarginLayoutParams) aVar3).leftMargin + ((ViewGroup.MarginLayoutParams) aVar3).rightMargin;
                            int measuredWidth2 = virtualChildAt4.getMeasuredWidth() + i49;
                            i40 = Math.max(i40, measuredWidth2);
                            float f8 = f2;
                            if (mode != 1073741824) {
                                i8 = i4;
                                i9 = -1;
                                if (((ViewGroup.MarginLayoutParams) aVar3).width == -1) {
                                    z = true;
                                    if (!z) {
                                        i49 = measuredWidth2;
                                    }
                                    int max4 = Math.max(i46, i49);
                                    boolean z7 = !z6 && ((ViewGroup.MarginLayoutParams) aVar3).width == i9;
                                    int i50 = this.mTotalLength;
                                    this.mTotalLength = Math.max(i50, getNextLocationOffset(virtualChildAt4) + virtualChildAt4.getMeasuredHeight() + i50 + ((ViewGroup.MarginLayoutParams) aVar3).topMargin + ((ViewGroup.MarginLayoutParams) aVar3).bottomMargin);
                                    z6 = z7;
                                    i4 = i8;
                                    i46 = max4;
                                    f2 = f8;
                                }
                            } else {
                                i8 = i4;
                                i9 = -1;
                            }
                            z = false;
                            if (!z) {
                            }
                            int max42 = Math.max(i46, i49);
                            if (z6) {
                            }
                            int i502 = this.mTotalLength;
                            this.mTotalLength = Math.max(i502, getNextLocationOffset(virtualChildAt4) + virtualChildAt4.getMeasuredHeight() + i502 + ((ViewGroup.MarginLayoutParams) aVar3).topMargin + ((ViewGroup.MarginLayoutParams) aVar3).bottomMargin);
                            z6 = z7;
                            i4 = i8;
                            i46 = max42;
                            f2 = f8;
                        }
                        i47++;
                        i45 = i7;
                    }
                    i5 = i;
                    this.mTotalLength = getPaddingBottom() + getPaddingTop() + this.mTotalLength;
                    i6 = i46;
                }
                if (z6 || mode == 1073741824) {
                    i6 = i40;
                }
                setMeasuredDimension(View.resolveSizeAndState(Math.max(getPaddingRight() + getPaddingLeft() + i6, getSuggestedMinimumWidth()), i5, i4), resolveSizeAndState);
                if (z5) {
                    forceUniformWidth(i3, i2);
                    return;
                }
                return;
            }
        }
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
        if (this.mDivider == null) {
            return;
        }
        if (this.mOrientation == 1) {
            drawDividersVertical(canvas);
        } else {
            drawDividersHorizontal(canvas);
        }
    }

    @Override // android.view.View
    public void onInitializeAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(accessibilityEvent);
        accessibilityEvent.setClassName(ACCESSIBILITY_CLASS_NAME);
    }

    @Override // android.view.View
    public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo accessibilityNodeInfo) {
        super.onInitializeAccessibilityNodeInfo(accessibilityNodeInfo);
        accessibilityNodeInfo.setClassName(ACCESSIBILITY_CLASS_NAME);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        if (this.mOrientation == 1) {
            layoutVertical(i, i2, i3, i4);
        } else {
            layoutHorizontal(i, i2, i3, i4);
        }
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        if (this.mOrientation == 1) {
            measureVertical(i, i2);
        } else {
            measureHorizontal(i, i2);
        }
    }

    public void setBaselineAligned(boolean z) {
        this.mBaselineAligned = z;
    }

    public void setBaselineAlignedChildIndex(int i) {
        if (i >= 0 && i < getChildCount()) {
            this.mBaselineAlignedChildIndex = i;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("base aligned child index out of range (0, ");
        x.append(getChildCount());
        x.append(")");
        throw new IllegalArgumentException(x.toString());
    }

    public void setDividerDrawable(Drawable drawable) {
        if (drawable == this.mDivider) {
            return;
        }
        this.mDivider = drawable;
        if (drawable != null) {
            this.mDividerWidth = drawable.getIntrinsicWidth();
            this.mDividerHeight = drawable.getIntrinsicHeight();
        } else {
            this.mDividerWidth = 0;
            this.mDividerHeight = 0;
        }
        setWillNotDraw(drawable == null);
        requestLayout();
    }

    public void setDividerPadding(int i) {
        this.mDividerPadding = i;
    }

    public void setGravity(int i) {
        if (this.mGravity != i) {
            if ((8388615 & i) == 0) {
                i |= 8388611;
            }
            if ((i & 112) == 0) {
                i |= 48;
            }
            this.mGravity = i;
            requestLayout();
        }
    }

    public void setHorizontalGravity(int i) {
        int i2 = i & 8388615;
        int i3 = this.mGravity;
        if ((8388615 & i3) != i2) {
            this.mGravity = i2 | ((-8388616) & i3);
            requestLayout();
        }
    }

    public void setMeasureWithLargestChildEnabled(boolean z) {
        this.mUseLargestChild = z;
    }

    public void setOrientation(int i) {
        if (this.mOrientation != i) {
            this.mOrientation = i;
            requestLayout();
        }
    }

    public void setShowDividers(int i) {
        if (i != this.mShowDividers) {
            requestLayout();
        }
        this.mShowDividers = i;
    }

    public void setVerticalGravity(int i) {
        int i2 = i & 112;
        int i3 = this.mGravity;
        if ((i3 & 112) != i2) {
            this.mGravity = i2 | (i3 & (-113));
            requestLayout();
        }
    }

    public void setWeightSum(float f2) {
        this.mWeightSum = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f2);
    }

    @Override // android.view.ViewGroup
    public boolean shouldDelayChildPressedState() {
        return false;
    }

    public i0(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.view.ViewGroup
    public a generateDefaultLayoutParams() {
        int i = this.mOrientation;
        if (i == 0) {
            return new a(-2, -2);
        }
        if (i == 1) {
            return new a(-1, -2);
        }
        return null;
    }

    public i0(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        Drawable drawable;
        int resourceId;
        this.mBaselineAligned = true;
        this.mBaselineAlignedChildIndex = -1;
        this.mBaselineChildTop = 0;
        this.mGravity = BadgeDrawable.TOP_START;
        int[] iArr = b.b.b.m;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, iArr, i, 0);
        b.j.j.q.m(this, context, iArr, attributeSet, obtainStyledAttributes, i, 0);
        int i2 = obtainStyledAttributes.getInt(1, -1);
        if (i2 >= 0) {
            setOrientation(i2);
        }
        int i3 = obtainStyledAttributes.getInt(0, -1);
        if (i3 >= 0) {
            setGravity(i3);
        }
        boolean z = obtainStyledAttributes.getBoolean(2, true);
        if (!z) {
            setBaselineAligned(z);
        }
        this.mWeightSum = obtainStyledAttributes.getFloat(4, -1.0f);
        this.mBaselineAlignedChildIndex = obtainStyledAttributes.getInt(3, -1);
        this.mUseLargestChild = obtainStyledAttributes.getBoolean(7, false);
        if (obtainStyledAttributes.hasValue(5) && (resourceId = obtainStyledAttributes.getResourceId(5, 0)) != 0) {
            drawable = b.b.d.a.a.a(context, resourceId);
        } else {
            drawable = obtainStyledAttributes.getDrawable(5);
        }
        setDividerDrawable(drawable);
        this.mShowDividers = obtainStyledAttributes.getInt(8, 0);
        this.mDividerPadding = obtainStyledAttributes.getDimensionPixelSize(6, 0);
        obtainStyledAttributes.recycle();
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.view.ViewGroup
    public a generateLayoutParams(AttributeSet attributeSet) {
        return new a(getContext(), attributeSet);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.view.ViewGroup
    public a generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return new a(layoutParams);
    }

    /* compiled from: LinearLayoutCompat.java */
    /* loaded from: classes.dex */
    public static class a extends ViewGroup.MarginLayoutParams {

        /* renamed from: a  reason: collision with root package name */
        public float f860a;

        /* renamed from: b  reason: collision with root package name */
        public int f861b;

        public a(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f861b = -1;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.n);
            this.f860a = obtainStyledAttributes.getFloat(3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.f861b = obtainStyledAttributes.getInt(0, -1);
            obtainStyledAttributes.recycle();
        }

        public a(int i, int i2) {
            super(i, i2);
            this.f861b = -1;
            this.f860a = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public a(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f861b = -1;
        }
    }
}