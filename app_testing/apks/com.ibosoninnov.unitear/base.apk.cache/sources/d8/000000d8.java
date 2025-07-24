package androidx.recyclerview.widget;

import android.animation.LayoutTransition;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.database.Observable;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.StateListDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.os.SystemClock;
import android.os.Trace;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseArray;
import android.view.Display;
import android.view.FocusFinder;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.ViewPropertyAnimator;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityManager;
import android.view.animation.Interpolator;
import android.widget.EdgeEffect;
import android.widget.OverScroller;
import b.j.j.x.b;
import b.w.b.a;
import b.w.b.b;
import b.w.b.k;
import b.w.b.m;
import b.w.b.v;
import b.w.b.y;
import b.w.b.z;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class RecyclerView extends ViewGroup implements b.j.j.e {
    public static final boolean DEBUG = false;
    public static final int DEFAULT_ORIENTATION = 1;
    public static final boolean DISPATCH_TEMP_DETACH = false;
    public static final long FOREVER_NS = Long.MAX_VALUE;
    public static final int HORIZONTAL = 0;
    private static final int INVALID_POINTER = -1;
    public static final int INVALID_TYPE = -1;
    private static final Class<?>[] LAYOUT_MANAGER_CONSTRUCTOR_SIGNATURE;
    public static final int MAX_SCROLL_DURATION = 2000;
    public static final long NO_ID = -1;
    public static final int NO_POSITION = -1;
    public static final int SCROLL_STATE_DRAGGING = 1;
    public static final int SCROLL_STATE_IDLE = 0;
    public static final int SCROLL_STATE_SETTLING = 2;
    public static final String TAG = "RecyclerView";
    public static final int TOUCH_SLOP_DEFAULT = 0;
    public static final int TOUCH_SLOP_PAGING = 1;
    public static final String TRACE_BIND_VIEW_TAG = "RV OnBindView";
    public static final String TRACE_CREATE_VIEW_TAG = "RV CreateView";
    private static final String TRACE_HANDLE_ADAPTER_UPDATES_TAG = "RV PartialInvalidate";
    public static final String TRACE_NESTED_PREFETCH_TAG = "RV Nested Prefetch";
    private static final String TRACE_ON_DATA_SET_CHANGE_LAYOUT_TAG = "RV FullInvalidate";
    private static final String TRACE_ON_LAYOUT_TAG = "RV OnLayout";
    public static final String TRACE_PREFETCH_TAG = "RV Prefetch";
    public static final String TRACE_SCROLL_TAG = "RV Scroll";
    public static final int UNDEFINED_DURATION = Integer.MIN_VALUE;
    public static final boolean VERBOSE_TRACING = false;
    public static final int VERTICAL = 1;
    public static final Interpolator sQuinticInterpolator;
    public b.w.b.v mAccessibilityDelegate;
    private final AccessibilityManager mAccessibilityManager;
    public g mAdapter;
    public b.w.b.a mAdapterHelper;
    public boolean mAdapterUpdateDuringMeasure;
    private EdgeEffect mBottomGlow;
    private j mChildDrawingOrderCallback;
    public b.w.b.b mChildHelper;
    public boolean mClipToPadding;
    public boolean mDataSetHasChangedAfterLayout;
    public boolean mDispatchItemsChangedEvent;
    private int mDispatchScrollCounter;
    private int mEatenAccessibilityChangeFlags;
    private k mEdgeEffectFactory;
    public boolean mEnableFastScroller;
    public boolean mFirstLayoutComplete;
    public b.w.b.m mGapWorker;
    public boolean mHasFixedSize;
    private boolean mIgnoreMotionEventTillDown;
    private int mInitialTouchX;
    private int mInitialTouchY;
    private int mInterceptRequestLayoutDepth;
    private s mInterceptingOnItemTouchListener;
    public boolean mIsAttached;
    public l mItemAnimator;
    private l.b mItemAnimatorListener;
    private Runnable mItemAnimatorRunner;
    public final ArrayList<n> mItemDecorations;
    public boolean mItemsAddedOrRemoved;
    public boolean mItemsChanged;
    private int mLastTouchX;
    private int mLastTouchY;
    public o mLayout;
    private int mLayoutOrScrollCounter;
    public boolean mLayoutSuppressed;
    public boolean mLayoutWasDefered;
    private EdgeEffect mLeftGlow;
    private final int mMaxFlingVelocity;
    private final int mMinFlingVelocity;
    private final int[] mMinMaxLayoutPositions;
    private final int[] mNestedOffsets;
    private final x mObserver;
    private List<q> mOnChildAttachStateListeners;
    private r mOnFlingListener;
    private final ArrayList<s> mOnItemTouchListeners;
    public final List<d0> mPendingAccessibilityImportanceChange;
    private y mPendingSavedState;
    public boolean mPostedAnimatorRunner;
    public m.b mPrefetchRegistry;
    private boolean mPreserveFocusAfterLayout;
    public final v mRecycler;
    public w mRecyclerListener;
    public final int[] mReusableIntPair;
    private EdgeEffect mRightGlow;
    private float mScaledHorizontalScrollFactor;
    private float mScaledVerticalScrollFactor;
    private t mScrollListener;
    private List<t> mScrollListeners;
    private final int[] mScrollOffset;
    private int mScrollPointerId;
    private int mScrollState;
    private b.j.j.f mScrollingChildHelper;
    public final a0 mState;
    public final Rect mTempRect;
    private final Rect mTempRect2;
    public final RectF mTempRectF;
    private EdgeEffect mTopGlow;
    private int mTouchSlop;
    public final Runnable mUpdateChildViewsRunnable;
    private VelocityTracker mVelocityTracker;
    public final c0 mViewFlinger;
    private final z.b mViewInfoProcessCallback;
    public final b.w.b.z mViewInfoStore;
    private static final int[] NESTED_SCROLLING_ATTRS = {16843830};
    public static final boolean FORCE_INVALIDATE_DISPLAY_LIST = false;
    public static final boolean ALLOW_SIZE_IN_UNSPECIFIED_SPEC = true;
    public static final boolean POST_UPDATES_ON_ANIMATION = true;
    public static final boolean ALLOW_THREAD_GAP_WORK = true;
    private static final boolean FORCE_ABS_FOCUS_SEARCH_DIRECTION = false;
    private static final boolean IGNORE_DETACHED_FOCUSED_CHILD = false;

    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            RecyclerView recyclerView = RecyclerView.this;
            if (!recyclerView.mFirstLayoutComplete || recyclerView.isLayoutRequested()) {
                return;
            }
            RecyclerView recyclerView2 = RecyclerView.this;
            if (!recyclerView2.mIsAttached) {
                recyclerView2.requestLayout();
            } else if (recyclerView2.mLayoutSuppressed) {
                recyclerView2.mLayoutWasDefered = true;
            } else {
                recyclerView2.consumePendingUpdateOperations();
            }
        }
    }

    /* loaded from: classes.dex */
    public static class a0 {

        /* renamed from: a  reason: collision with root package name */
        public int f390a = -1;

        /* renamed from: b  reason: collision with root package name */
        public int f391b = 0;

        /* renamed from: c  reason: collision with root package name */
        public int f392c = 0;

        /* renamed from: d  reason: collision with root package name */
        public int f393d = 1;

        /* renamed from: e  reason: collision with root package name */
        public int f394e = 0;

        /* renamed from: f  reason: collision with root package name */
        public boolean f395f = false;

        /* renamed from: g  reason: collision with root package name */
        public boolean f396g = false;

        /* renamed from: h  reason: collision with root package name */
        public boolean f397h = false;
        public boolean i = false;
        public boolean j = false;
        public boolean k = false;
        public int l;
        public long m;
        public int n;

        public void a(int i) {
            if ((this.f393d & i) != 0) {
                return;
            }
            StringBuilder x = c.b.a.a.a.x("Layout state should be one of ");
            x.append(Integer.toBinaryString(i));
            x.append(" but it is ");
            x.append(Integer.toBinaryString(this.f393d));
            throw new IllegalStateException(x.toString());
        }

        public int b() {
            return this.f396g ? this.f391b - this.f392c : this.f394e;
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("State{mTargetPosition=");
            x.append(this.f390a);
            x.append(", mData=");
            x.append((Object) null);
            x.append(", mItemCount=");
            x.append(this.f394e);
            x.append(", mIsMeasuring=");
            x.append(this.i);
            x.append(", mPreviousLayoutItemCount=");
            x.append(this.f391b);
            x.append(", mDeletedInvisibleItemCountSincePreviousLayout=");
            x.append(this.f392c);
            x.append(", mStructureChanged=");
            x.append(this.f395f);
            x.append(", mInPreLayout=");
            x.append(this.f396g);
            x.append(", mRunSimpleAnimations=");
            x.append(this.j);
            x.append(", mRunPredictiveAnimations=");
            x.append(this.k);
            x.append('}');
            return x.toString();
        }
    }

    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            l lVar = RecyclerView.this.mItemAnimator;
            if (lVar != null) {
                b.w.b.k kVar = (b.w.b.k) lVar;
                boolean z = !kVar.i.isEmpty();
                boolean z2 = !kVar.k.isEmpty();
                boolean z3 = !kVar.l.isEmpty();
                boolean z4 = !kVar.j.isEmpty();
                if (z || z2 || z4 || z3) {
                    Iterator<d0> it = kVar.i.iterator();
                    while (it.hasNext()) {
                        d0 next = it.next();
                        View view = next.itemView;
                        ViewPropertyAnimator animate = view.animate();
                        kVar.r.add(next);
                        animate.setDuration(kVar.f412d).alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setListener(new b.w.b.f(kVar, next, animate, view)).start();
                    }
                    kVar.i.clear();
                    if (z2) {
                        ArrayList<k.b> arrayList = new ArrayList<>();
                        arrayList.addAll(kVar.k);
                        kVar.n.add(arrayList);
                        kVar.k.clear();
                        b.w.b.c cVar = new b.w.b.c(kVar, arrayList);
                        if (z) {
                            View view2 = arrayList.get(0).f2752a.itemView;
                            long j = kVar.f412d;
                            AtomicInteger atomicInteger = b.j.j.q.f2214a;
                            view2.postOnAnimationDelayed(cVar, j);
                        } else {
                            cVar.run();
                        }
                    }
                    if (z3) {
                        ArrayList<k.a> arrayList2 = new ArrayList<>();
                        arrayList2.addAll(kVar.l);
                        kVar.o.add(arrayList2);
                        kVar.l.clear();
                        b.w.b.d dVar = new b.w.b.d(kVar, arrayList2);
                        if (z) {
                            View view3 = arrayList2.get(0).f2746a.itemView;
                            long j2 = kVar.f412d;
                            AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                            view3.postOnAnimationDelayed(dVar, j2);
                        } else {
                            dVar.run();
                        }
                    }
                    if (z4) {
                        ArrayList<d0> arrayList3 = new ArrayList<>();
                        arrayList3.addAll(kVar.j);
                        kVar.m.add(arrayList3);
                        kVar.j.clear();
                        b.w.b.e eVar = new b.w.b.e(kVar, arrayList3);
                        if (!z && !z2 && !z3) {
                            eVar.run();
                        } else {
                            long j3 = z ? kVar.f412d : 0L;
                            long j4 = z2 ? kVar.f413e : 0L;
                            long j5 = z3 ? kVar.f414f : 0L;
                            View view4 = arrayList3.get(0).itemView;
                            AtomicInteger atomicInteger3 = b.j.j.q.f2214a;
                            view4.postOnAnimationDelayed(eVar, Math.max(j4, j5) + j3);
                        }
                    }
                }
            }
            RecyclerView.this.mPostedAnimatorRunner = false;
        }
    }

    /* loaded from: classes.dex */
    public static abstract class b0 {
    }

    /* loaded from: classes.dex */
    public static class c implements Interpolator {
        @Override // android.animation.TimeInterpolator
        public float getInterpolation(float f2) {
            float f3 = f2 - 1.0f;
            return (f3 * f3 * f3 * f3 * f3) + 1.0f;
        }
    }

    /* loaded from: classes.dex */
    public class c0 implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public int f399b;

        /* renamed from: c  reason: collision with root package name */
        public int f400c;

        /* renamed from: d  reason: collision with root package name */
        public OverScroller f401d;

        /* renamed from: e  reason: collision with root package name */
        public Interpolator f402e;

        /* renamed from: f  reason: collision with root package name */
        public boolean f403f;

        /* renamed from: g  reason: collision with root package name */
        public boolean f404g;

        public c0() {
            Interpolator interpolator = RecyclerView.sQuinticInterpolator;
            this.f402e = interpolator;
            this.f403f = false;
            this.f404g = false;
            this.f401d = new OverScroller(RecyclerView.this.getContext(), interpolator);
        }

        public void a() {
            if (this.f403f) {
                this.f404g = true;
                return;
            }
            RecyclerView.this.removeCallbacks(this);
            RecyclerView recyclerView = RecyclerView.this;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            recyclerView.postOnAnimation(this);
        }

        public void b(int i, int i2, int i3, Interpolator interpolator) {
            int i4;
            if (i3 == Integer.MIN_VALUE) {
                int abs = Math.abs(i);
                int abs2 = Math.abs(i2);
                boolean z = abs > abs2;
                int sqrt = (int) Math.sqrt(0);
                int sqrt2 = (int) Math.sqrt((i2 * i2) + (i * i));
                RecyclerView recyclerView = RecyclerView.this;
                int width = z ? recyclerView.getWidth() : recyclerView.getHeight();
                int i5 = width / 2;
                float f2 = width;
                float f3 = i5;
                float sin = (((float) Math.sin((Math.min(1.0f, (sqrt2 * 1.0f) / f2) - 0.5f) * 0.47123894f)) * f3) + f3;
                if (sqrt > 0) {
                    i4 = Math.round(Math.abs(sin / sqrt) * 1000.0f) * 4;
                } else {
                    if (!z) {
                        abs = abs2;
                    }
                    i4 = (int) (((abs / f2) + 1.0f) * 300.0f);
                }
                i3 = Math.min(i4, 2000);
            }
            int i6 = i3;
            if (interpolator == null) {
                interpolator = RecyclerView.sQuinticInterpolator;
            }
            if (this.f402e != interpolator) {
                this.f402e = interpolator;
                this.f401d = new OverScroller(RecyclerView.this.getContext(), interpolator);
            }
            this.f400c = 0;
            this.f399b = 0;
            RecyclerView.this.setScrollState(2);
            this.f401d.startScroll(0, 0, i, i2, i6);
            a();
        }

        public void c() {
            RecyclerView.this.removeCallbacks(this);
            this.f401d.abortAnimation();
        }

        @Override // java.lang.Runnable
        public void run() {
            int i;
            int i2;
            RecyclerView recyclerView = RecyclerView.this;
            if (recyclerView.mLayout == null) {
                c();
                return;
            }
            this.f404g = false;
            this.f403f = true;
            recyclerView.consumePendingUpdateOperations();
            OverScroller overScroller = this.f401d;
            if (overScroller.computeScrollOffset()) {
                int currX = overScroller.getCurrX();
                int currY = overScroller.getCurrY();
                int i3 = currX - this.f399b;
                int i4 = currY - this.f400c;
                this.f399b = currX;
                this.f400c = currY;
                RecyclerView recyclerView2 = RecyclerView.this;
                int[] iArr = recyclerView2.mReusableIntPair;
                iArr[0] = 0;
                iArr[1] = 0;
                if (recyclerView2.dispatchNestedPreScroll(i3, i4, iArr, null, 1)) {
                    int[] iArr2 = RecyclerView.this.mReusableIntPair;
                    i3 -= iArr2[0];
                    i4 -= iArr2[1];
                }
                if (RecyclerView.this.getOverScrollMode() != 2) {
                    RecyclerView.this.considerReleasingGlowsOnScroll(i3, i4);
                }
                RecyclerView recyclerView3 = RecyclerView.this;
                if (recyclerView3.mAdapter != null) {
                    int[] iArr3 = recyclerView3.mReusableIntPair;
                    iArr3[0] = 0;
                    iArr3[1] = 0;
                    recyclerView3.scrollStep(i3, i4, iArr3);
                    RecyclerView recyclerView4 = RecyclerView.this;
                    int[] iArr4 = recyclerView4.mReusableIntPair;
                    i2 = iArr4[0];
                    i = iArr4[1];
                    i3 -= i2;
                    i4 -= i;
                    z zVar = recyclerView4.mLayout.mSmoothScroller;
                    if (zVar != null && !zVar.isPendingInitialRun() && zVar.isRunning()) {
                        int b2 = RecyclerView.this.mState.b();
                        if (b2 == 0) {
                            zVar.stop();
                        } else if (zVar.getTargetPosition() >= b2) {
                            zVar.setTargetPosition(b2 - 1);
                            zVar.onAnimation(i2, i);
                        } else {
                            zVar.onAnimation(i2, i);
                        }
                    }
                } else {
                    i = 0;
                    i2 = 0;
                }
                if (!RecyclerView.this.mItemDecorations.isEmpty()) {
                    RecyclerView.this.invalidate();
                }
                RecyclerView recyclerView5 = RecyclerView.this;
                int[] iArr5 = recyclerView5.mReusableIntPair;
                iArr5[0] = 0;
                iArr5[1] = 0;
                recyclerView5.dispatchNestedScroll(i2, i, i3, i4, null, 1, iArr5);
                RecyclerView recyclerView6 = RecyclerView.this;
                int[] iArr6 = recyclerView6.mReusableIntPair;
                int i5 = i3 - iArr6[0];
                int i6 = i4 - iArr6[1];
                if (i2 != 0 || i != 0) {
                    recyclerView6.dispatchOnScrolled(i2, i);
                }
                if (!RecyclerView.this.awakenScrollBars()) {
                    RecyclerView.this.invalidate();
                }
                boolean z = overScroller.isFinished() || (((overScroller.getCurrX() == overScroller.getFinalX()) || i5 != 0) && ((overScroller.getCurrY() == overScroller.getFinalY()) || i6 != 0));
                z zVar2 = RecyclerView.this.mLayout.mSmoothScroller;
                if (!(zVar2 != null && zVar2.isPendingInitialRun()) && z) {
                    if (RecyclerView.this.getOverScrollMode() != 2) {
                        int currVelocity = (int) overScroller.getCurrVelocity();
                        int i7 = i5 < 0 ? -currVelocity : i5 > 0 ? currVelocity : 0;
                        if (i6 < 0) {
                            currVelocity = -currVelocity;
                        } else if (i6 <= 0) {
                            currVelocity = 0;
                        }
                        RecyclerView.this.absorbGlows(i7, currVelocity);
                    }
                    if (RecyclerView.ALLOW_THREAD_GAP_WORK) {
                        m.b bVar = RecyclerView.this.mPrefetchRegistry;
                        int[] iArr7 = bVar.f2778c;
                        if (iArr7 != null) {
                            Arrays.fill(iArr7, -1);
                        }
                        bVar.f2779d = 0;
                    }
                } else {
                    a();
                    RecyclerView recyclerView7 = RecyclerView.this;
                    b.w.b.m mVar = recyclerView7.mGapWorker;
                    if (mVar != null) {
                        mVar.a(recyclerView7, i2, i);
                    }
                }
            }
            z zVar3 = RecyclerView.this.mLayout.mSmoothScroller;
            if (zVar3 != null && zVar3.isPendingInitialRun()) {
                zVar3.onAnimation(0, 0);
            }
            this.f403f = false;
            if (this.f404g) {
                RecyclerView.this.removeCallbacks(this);
                RecyclerView recyclerView8 = RecyclerView.this;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                recyclerView8.postOnAnimation(this);
                return;
            }
            RecyclerView.this.setScrollState(0);
            RecyclerView.this.stopNestedScroll(1);
        }
    }

    /* loaded from: classes.dex */
    public class d implements z.b {
        public d() {
        }
    }

    /* loaded from: classes.dex */
    public static abstract class d0 {
        public static final int FLAG_ADAPTER_FULLUPDATE = 1024;
        public static final int FLAG_ADAPTER_POSITION_UNKNOWN = 512;
        public static final int FLAG_APPEARED_IN_PRE_LAYOUT = 4096;
        public static final int FLAG_BOUNCED_FROM_HIDDEN_LIST = 8192;
        public static final int FLAG_BOUND = 1;
        public static final int FLAG_IGNORE = 128;
        public static final int FLAG_INVALID = 4;
        public static final int FLAG_MOVED = 2048;
        public static final int FLAG_NOT_RECYCLABLE = 16;
        public static final int FLAG_REMOVED = 8;
        public static final int FLAG_RETURNED_FROM_SCRAP = 32;
        public static final int FLAG_TMP_DETACHED = 256;
        public static final int FLAG_UPDATE = 2;
        private static final List<Object> FULLUPDATE_PAYLOADS = Collections.emptyList();
        public static final int PENDING_ACCESSIBILITY_STATE_NOT_SET = -1;
        public final View itemView;
        public int mFlags;
        public WeakReference<RecyclerView> mNestedRecyclerView;
        public RecyclerView mOwnerRecyclerView;
        public int mPosition = -1;
        public int mOldPosition = -1;
        public long mItemId = -1;
        public int mItemViewType = -1;
        public int mPreLayoutPosition = -1;
        public d0 mShadowedHolder = null;
        public d0 mShadowingHolder = null;
        public List<Object> mPayloads = null;
        public List<Object> mUnmodifiedPayloads = null;
        private int mIsRecyclableCount = 0;
        public v mScrapContainer = null;
        public boolean mInChangeScrap = false;
        private int mWasImportantForAccessibilityBeforeHidden = 0;
        public int mPendingAccessibilityState = -1;

        public d0(View view) {
            if (view != null) {
                this.itemView = view;
                return;
            }
            throw new IllegalArgumentException("itemView may not be null");
        }

        private void createPayloadsIfNeeded() {
            if (this.mPayloads == null) {
                ArrayList arrayList = new ArrayList();
                this.mPayloads = arrayList;
                this.mUnmodifiedPayloads = Collections.unmodifiableList(arrayList);
            }
        }

        public void addChangePayload(Object obj) {
            if (obj == null) {
                addFlags(1024);
            } else if ((1024 & this.mFlags) == 0) {
                createPayloadsIfNeeded();
                this.mPayloads.add(obj);
            }
        }

        public void addFlags(int i) {
            this.mFlags = i | this.mFlags;
        }

        public void clearOldPosition() {
            this.mOldPosition = -1;
            this.mPreLayoutPosition = -1;
        }

        public void clearPayload() {
            List<Object> list = this.mPayloads;
            if (list != null) {
                list.clear();
            }
            this.mFlags &= -1025;
        }

        public void clearReturnedFromScrapFlag() {
            this.mFlags &= -33;
        }

        public void clearTmpDetachFlag() {
            this.mFlags &= -257;
        }

        public boolean doesTransientStatePreventRecycling() {
            if ((this.mFlags & 16) == 0) {
                View view = this.itemView;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                if (view.hasTransientState()) {
                    return true;
                }
            }
            return false;
        }

        public void flagRemovedAndOffsetPosition(int i, int i2, boolean z) {
            addFlags(8);
            offsetPosition(i2, z);
            this.mPosition = i;
        }

        public final int getAdapterPosition() {
            RecyclerView recyclerView = this.mOwnerRecyclerView;
            if (recyclerView == null) {
                return -1;
            }
            return recyclerView.getAdapterPositionFor(this);
        }

        public final long getItemId() {
            return this.mItemId;
        }

        public final int getItemViewType() {
            return this.mItemViewType;
        }

        public final int getLayoutPosition() {
            int i = this.mPreLayoutPosition;
            return i == -1 ? this.mPosition : i;
        }

        public final int getOldPosition() {
            return this.mOldPosition;
        }

        @Deprecated
        public final int getPosition() {
            int i = this.mPreLayoutPosition;
            return i == -1 ? this.mPosition : i;
        }

        public List<Object> getUnmodifiedPayloads() {
            if ((this.mFlags & 1024) == 0) {
                List<Object> list = this.mPayloads;
                if (list != null && list.size() != 0) {
                    return this.mUnmodifiedPayloads;
                }
                return FULLUPDATE_PAYLOADS;
            }
            return FULLUPDATE_PAYLOADS;
        }

        public boolean hasAnyOfTheFlags(int i) {
            return (i & this.mFlags) != 0;
        }

        public boolean isAdapterPositionUnknown() {
            return (this.mFlags & 512) != 0 || isInvalid();
        }

        public boolean isAttachedToTransitionOverlay() {
            return (this.itemView.getParent() == null || this.itemView.getParent() == this.mOwnerRecyclerView) ? false : true;
        }

        public boolean isBound() {
            return (this.mFlags & 1) != 0;
        }

        public boolean isInvalid() {
            return (this.mFlags & 4) != 0;
        }

        public final boolean isRecyclable() {
            if ((this.mFlags & 16) == 0) {
                View view = this.itemView;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                if (!view.hasTransientState()) {
                    return true;
                }
            }
            return false;
        }

        public boolean isRemoved() {
            return (this.mFlags & 8) != 0;
        }

        public boolean isScrap() {
            return this.mScrapContainer != null;
        }

        public boolean isTmpDetached() {
            return (this.mFlags & 256) != 0;
        }

        public boolean isUpdated() {
            return (this.mFlags & 2) != 0;
        }

        public boolean needsUpdate() {
            return (this.mFlags & 2) != 0;
        }

        public void offsetPosition(int i, boolean z) {
            if (this.mOldPosition == -1) {
                this.mOldPosition = this.mPosition;
            }
            if (this.mPreLayoutPosition == -1) {
                this.mPreLayoutPosition = this.mPosition;
            }
            if (z) {
                this.mPreLayoutPosition += i;
            }
            this.mPosition += i;
            if (this.itemView.getLayoutParams() != null) {
                ((p) this.itemView.getLayoutParams()).f426c = true;
            }
        }

        public void onEnteredHiddenState(RecyclerView recyclerView) {
            int i = this.mPendingAccessibilityState;
            if (i != -1) {
                this.mWasImportantForAccessibilityBeforeHidden = i;
            } else {
                View view = this.itemView;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                this.mWasImportantForAccessibilityBeforeHidden = view.getImportantForAccessibility();
            }
            recyclerView.setChildImportantForAccessibilityInternal(this, 4);
        }

        public void onLeftHiddenState(RecyclerView recyclerView) {
            recyclerView.setChildImportantForAccessibilityInternal(this, this.mWasImportantForAccessibilityBeforeHidden);
            this.mWasImportantForAccessibilityBeforeHidden = 0;
        }

        public void resetInternal() {
            this.mFlags = 0;
            this.mPosition = -1;
            this.mOldPosition = -1;
            this.mItemId = -1L;
            this.mPreLayoutPosition = -1;
            this.mIsRecyclableCount = 0;
            this.mShadowedHolder = null;
            this.mShadowingHolder = null;
            clearPayload();
            this.mWasImportantForAccessibilityBeforeHidden = 0;
            this.mPendingAccessibilityState = -1;
            RecyclerView.clearNestedRecyclerViewIfNotNested(this);
        }

        public void saveOldPosition() {
            if (this.mOldPosition == -1) {
                this.mOldPosition = this.mPosition;
            }
        }

        public void setFlags(int i, int i2) {
            this.mFlags = (i & i2) | (this.mFlags & (~i2));
        }

        public final void setIsRecyclable(boolean z) {
            int i = this.mIsRecyclableCount;
            int i2 = z ? i - 1 : i + 1;
            this.mIsRecyclableCount = i2;
            if (i2 < 0) {
                this.mIsRecyclableCount = 0;
                Log.e("View", "isRecyclable decremented below 0: unmatched pair of setIsRecyable() calls for " + this);
            } else if (!z && i2 == 1) {
                this.mFlags |= 16;
            } else if (z && i2 == 0) {
                this.mFlags &= -17;
            }
        }

        public void setScrapContainer(v vVar, boolean z) {
            this.mScrapContainer = vVar;
            this.mInChangeScrap = z;
        }

        public boolean shouldBeKeptAsChild() {
            return (this.mFlags & 16) != 0;
        }

        public boolean shouldIgnore() {
            return (this.mFlags & 128) != 0;
        }

        public void stopIgnoring() {
            this.mFlags &= -129;
        }

        public String toString() {
            StringBuilder A = c.b.a.a.a.A(getClass().isAnonymousClass() ? "ViewHolder" : getClass().getSimpleName(), "{");
            A.append(Integer.toHexString(hashCode()));
            A.append(" position=");
            A.append(this.mPosition);
            A.append(" id=");
            A.append(this.mItemId);
            A.append(", oldPos=");
            A.append(this.mOldPosition);
            A.append(", pLpos:");
            A.append(this.mPreLayoutPosition);
            StringBuilder sb = new StringBuilder(A.toString());
            if (isScrap()) {
                sb.append(" scrap ");
                sb.append(this.mInChangeScrap ? "[changeScrap]" : "[attachedScrap]");
            }
            if (isInvalid()) {
                sb.append(" invalid");
            }
            if (!isBound()) {
                sb.append(" unbound");
            }
            if (needsUpdate()) {
                sb.append(" update");
            }
            if (isRemoved()) {
                sb.append(" removed");
            }
            if (shouldIgnore()) {
                sb.append(" ignored");
            }
            if (isTmpDetached()) {
                sb.append(" tmpDetached");
            }
            if (!isRecyclable()) {
                StringBuilder x = c.b.a.a.a.x(" not recyclable(");
                x.append(this.mIsRecyclableCount);
                x.append(")");
                sb.append(x.toString());
            }
            if (isAdapterPositionUnknown()) {
                sb.append(" undefined adapter position");
            }
            if (this.itemView.getParent() == null) {
                sb.append(" no parent");
            }
            sb.append("}");
            return sb.toString();
        }

        public void unScrap() {
            this.mScrapContainer.l(this);
        }

        public boolean wasReturnedFromScrap() {
            return (this.mFlags & 32) != 0;
        }
    }

    /* loaded from: classes.dex */
    public class e implements b.InterfaceC0054b {
        public e() {
        }

        public View a(int i) {
            return RecyclerView.this.getChildAt(i);
        }

        public int b() {
            return RecyclerView.this.getChildCount();
        }

        public void c(int i) {
            View childAt = RecyclerView.this.getChildAt(i);
            if (childAt != null) {
                RecyclerView.this.dispatchChildDetached(childAt);
                childAt.clearAnimation();
            }
            RecyclerView.this.removeViewAt(i);
        }
    }

    /* loaded from: classes.dex */
    public class f implements a.InterfaceC0053a {
        public f() {
        }

        public void a(a.b bVar) {
            int i = bVar.f2708a;
            if (i == 1) {
                RecyclerView recyclerView = RecyclerView.this;
                recyclerView.mLayout.onItemsAdded(recyclerView, bVar.f2709b, bVar.f2711d);
            } else if (i == 2) {
                RecyclerView recyclerView2 = RecyclerView.this;
                recyclerView2.mLayout.onItemsRemoved(recyclerView2, bVar.f2709b, bVar.f2711d);
            } else if (i == 4) {
                RecyclerView recyclerView3 = RecyclerView.this;
                recyclerView3.mLayout.onItemsUpdated(recyclerView3, bVar.f2709b, bVar.f2711d, bVar.f2710c);
            } else if (i != 8) {
            } else {
                RecyclerView recyclerView4 = RecyclerView.this;
                recyclerView4.mLayout.onItemsMoved(recyclerView4, bVar.f2709b, bVar.f2711d, 1);
            }
        }

        public d0 b(int i) {
            d0 findViewHolderForPosition = RecyclerView.this.findViewHolderForPosition(i, true);
            if (findViewHolderForPosition == null || RecyclerView.this.mChildHelper.k(findViewHolderForPosition.itemView)) {
                return null;
            }
            return findViewHolderForPosition;
        }

        public void c(int i, int i2, Object obj) {
            RecyclerView.this.viewRangeUpdate(i, i2, obj);
            RecyclerView.this.mItemsChanged = true;
        }
    }

    /* loaded from: classes.dex */
    public static class h extends Observable<i> {
        public boolean a() {
            return !((Observable) this).mObservers.isEmpty();
        }

        public void b() {
            for (int size = ((Observable) this).mObservers.size() - 1; size >= 0; size--) {
                ((i) ((Observable) this).mObservers.get(size)).onChanged();
            }
        }

        public void c(int i, int i2) {
            for (int size = ((Observable) this).mObservers.size() - 1; size >= 0; size--) {
                ((i) ((Observable) this).mObservers.get(size)).onItemRangeMoved(i, i2, 1);
            }
        }

        public void d(int i, int i2, Object obj) {
            for (int size = ((Observable) this).mObservers.size() - 1; size >= 0; size--) {
                ((i) ((Observable) this).mObservers.get(size)).onItemRangeChanged(i, i2, obj);
            }
        }

        public void e(int i, int i2) {
            for (int size = ((Observable) this).mObservers.size() - 1; size >= 0; size--) {
                ((i) ((Observable) this).mObservers.get(size)).onItemRangeInserted(i, i2);
            }
        }

        public void f(int i, int i2) {
            for (int size = ((Observable) this).mObservers.size() - 1; size >= 0; size--) {
                ((i) ((Observable) this).mObservers.get(size)).onItemRangeRemoved(i, i2);
            }
        }
    }

    /* loaded from: classes.dex */
    public static abstract class i {
        public void onChanged() {
        }

        public void onItemRangeChanged(int i, int i2) {
        }

        public void onItemRangeChanged(int i, int i2, Object obj) {
            onItemRangeChanged(i, i2);
        }

        public void onItemRangeInserted(int i, int i2) {
        }

        public void onItemRangeMoved(int i, int i2, int i3) {
        }

        public void onItemRangeRemoved(int i, int i2) {
        }
    }

    /* loaded from: classes.dex */
    public interface j {
        int a(int i, int i2);
    }

    /* loaded from: classes.dex */
    public static class k {
        public EdgeEffect a(RecyclerView recyclerView) {
            return new EdgeEffect(recyclerView.getContext());
        }
    }

    /* loaded from: classes.dex */
    public static abstract class l {

        /* renamed from: a  reason: collision with root package name */
        public b f409a = null;

        /* renamed from: b  reason: collision with root package name */
        public ArrayList<a> f410b = new ArrayList<>();

        /* renamed from: c  reason: collision with root package name */
        public long f411c = 120;

        /* renamed from: d  reason: collision with root package name */
        public long f412d = 120;

        /* renamed from: e  reason: collision with root package name */
        public long f413e = 250;

        /* renamed from: f  reason: collision with root package name */
        public long f414f = 250;

        /* loaded from: classes.dex */
        public interface a {
            void a();
        }

        /* loaded from: classes.dex */
        public interface b {
        }

        /* loaded from: classes.dex */
        public static class c {

            /* renamed from: a  reason: collision with root package name */
            public int f415a;

            /* renamed from: b  reason: collision with root package name */
            public int f416b;
        }

        public static int b(d0 d0Var) {
            int i = d0Var.mFlags & 14;
            if (d0Var.isInvalid()) {
                return 4;
            }
            if ((i & 4) == 0) {
                int oldPosition = d0Var.getOldPosition();
                int adapterPosition = d0Var.getAdapterPosition();
                return (oldPosition == -1 || adapterPosition == -1 || oldPosition == adapterPosition) ? i : i | 2048;
            }
            return i;
        }

        public abstract boolean a(d0 d0Var, d0 d0Var2, c cVar, c cVar2);

        public final void c(d0 d0Var) {
            b bVar = this.f409a;
            if (bVar != null) {
                m mVar = (m) bVar;
                Objects.requireNonNull(mVar);
                d0Var.setIsRecyclable(true);
                if (d0Var.mShadowedHolder != null && d0Var.mShadowingHolder == null) {
                    d0Var.mShadowedHolder = null;
                }
                d0Var.mShadowingHolder = null;
                if (d0Var.shouldBeKeptAsChild() || RecyclerView.this.removeAnimatingView(d0Var.itemView) || !d0Var.isTmpDetached()) {
                    return;
                }
                RecyclerView.this.removeDetachedView(d0Var.itemView, false);
            }
        }

        public final void d() {
            int size = this.f410b.size();
            for (int i = 0; i < size; i++) {
                this.f410b.get(i).a();
            }
            this.f410b.clear();
        }

        public abstract void e(d0 d0Var);

        public abstract void f();

        public abstract boolean g();

        /* JADX DEBUG: Incorrect args count in method signature: (Landroidx/recyclerview/widget/RecyclerView$a0;Landroidx/recyclerview/widget/RecyclerView$d0;ILjava/util/List<Ljava/lang/Object;>;)Landroidx/recyclerview/widget/RecyclerView$l$c; */
        public c h(d0 d0Var) {
            c cVar = new c();
            View view = d0Var.itemView;
            cVar.f415a = view.getLeft();
            cVar.f416b = view.getTop();
            view.getRight();
            view.getBottom();
            return cVar;
        }
    }

    /* loaded from: classes.dex */
    public class m implements l.b {
        public m() {
        }
    }

    /* loaded from: classes.dex */
    public static abstract class n {
        @Deprecated
        public void getItemOffsets(Rect rect, int i, RecyclerView recyclerView) {
            rect.set(0, 0, 0, 0);
        }

        @Deprecated
        public void onDraw(Canvas canvas, RecyclerView recyclerView) {
        }

        public void onDraw(Canvas canvas, RecyclerView recyclerView, a0 a0Var) {
            onDraw(canvas, recyclerView);
        }

        @Deprecated
        public void onDrawOver(Canvas canvas, RecyclerView recyclerView) {
        }

        public void onDrawOver(Canvas canvas, RecyclerView recyclerView, a0 a0Var) {
            onDrawOver(canvas, recyclerView);
        }

        public void getItemOffsets(Rect rect, View view, RecyclerView recyclerView, a0 a0Var) {
            getItemOffsets(rect, ((p) view.getLayoutParams()).a(), recyclerView);
        }
    }

    /* loaded from: classes.dex */
    public static abstract class o {
        public boolean mAutoMeasure;
        public b.w.b.b mChildHelper;
        private int mHeight;
        private int mHeightMode;
        public b.w.b.y mHorizontalBoundCheck;
        private final y.b mHorizontalBoundCheckCallback;
        public boolean mIsAttachedToWindow;
        private boolean mItemPrefetchEnabled;
        private boolean mMeasurementCacheEnabled;
        public int mPrefetchMaxCountObserved;
        public boolean mPrefetchMaxObservedInInitialPrefetch;
        public RecyclerView mRecyclerView;
        public boolean mRequestedSimpleAnimations;
        public z mSmoothScroller;
        public b.w.b.y mVerticalBoundCheck;
        private final y.b mVerticalBoundCheckCallback;
        private int mWidth;
        private int mWidthMode;

        /* loaded from: classes.dex */
        public class a implements y.b {
            public a() {
            }

            @Override // b.w.b.y.b
            public int a(View view) {
                return o.this.getDecoratedLeft(view) - ((ViewGroup.MarginLayoutParams) ((p) view.getLayoutParams())).leftMargin;
            }

            @Override // b.w.b.y.b
            public int b() {
                return o.this.getPaddingLeft();
            }

            @Override // b.w.b.y.b
            public int c() {
                return o.this.getWidth() - o.this.getPaddingRight();
            }

            @Override // b.w.b.y.b
            public View d(int i) {
                return o.this.getChildAt(i);
            }

            @Override // b.w.b.y.b
            public int e(View view) {
                return o.this.getDecoratedRight(view) + ((ViewGroup.MarginLayoutParams) ((p) view.getLayoutParams())).rightMargin;
            }
        }

        /* loaded from: classes.dex */
        public class b implements y.b {
            public b() {
            }

            @Override // b.w.b.y.b
            public int a(View view) {
                return o.this.getDecoratedTop(view) - ((ViewGroup.MarginLayoutParams) ((p) view.getLayoutParams())).topMargin;
            }

            @Override // b.w.b.y.b
            public int b() {
                return o.this.getPaddingTop();
            }

            @Override // b.w.b.y.b
            public int c() {
                return o.this.getHeight() - o.this.getPaddingBottom();
            }

            @Override // b.w.b.y.b
            public View d(int i) {
                return o.this.getChildAt(i);
            }

            @Override // b.w.b.y.b
            public int e(View view) {
                return o.this.getDecoratedBottom(view) + ((ViewGroup.MarginLayoutParams) ((p) view.getLayoutParams())).bottomMargin;
            }
        }

        /* loaded from: classes.dex */
        public interface c {
        }

        /* loaded from: classes.dex */
        public static class d {

            /* renamed from: a  reason: collision with root package name */
            public int f420a;

            /* renamed from: b  reason: collision with root package name */
            public int f421b;

            /* renamed from: c  reason: collision with root package name */
            public boolean f422c;

            /* renamed from: d  reason: collision with root package name */
            public boolean f423d;
        }

        public o() {
            a aVar = new a();
            this.mHorizontalBoundCheckCallback = aVar;
            b bVar = new b();
            this.mVerticalBoundCheckCallback = bVar;
            this.mHorizontalBoundCheck = new b.w.b.y(aVar);
            this.mVerticalBoundCheck = new b.w.b.y(bVar);
            this.mRequestedSimpleAnimations = false;
            this.mIsAttachedToWindow = false;
            this.mAutoMeasure = false;
            this.mMeasurementCacheEnabled = true;
            this.mItemPrefetchEnabled = true;
        }

        private void addViewInt(View view, int i, boolean z) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (!z && !childViewHolderInt.isRemoved()) {
                this.mRecyclerView.mViewInfoStore.f(childViewHolderInt);
            } else {
                this.mRecyclerView.mViewInfoStore.a(childViewHolderInt);
            }
            p pVar = (p) view.getLayoutParams();
            if (!childViewHolderInt.wasReturnedFromScrap() && !childViewHolderInt.isScrap()) {
                if (view.getParent() == this.mRecyclerView) {
                    int j = this.mChildHelper.j(view);
                    if (i == -1) {
                        i = this.mChildHelper.e();
                    }
                    if (j == -1) {
                        StringBuilder x = c.b.a.a.a.x("Added View has RecyclerView as parent but view is not a real child. Unfiltered index:");
                        x.append(this.mRecyclerView.indexOfChild(view));
                        throw new IllegalStateException(c.b.a.a.a.i(this.mRecyclerView, x));
                    } else if (j != i) {
                        this.mRecyclerView.mLayout.moveView(j, i);
                    }
                } else {
                    this.mChildHelper.a(view, i, false);
                    pVar.f426c = true;
                    z zVar = this.mSmoothScroller;
                    if (zVar != null && zVar.isRunning()) {
                        this.mSmoothScroller.onChildAttachedToWindow(view);
                    }
                }
            } else {
                if (childViewHolderInt.isScrap()) {
                    childViewHolderInt.unScrap();
                } else {
                    childViewHolderInt.clearReturnedFromScrapFlag();
                }
                this.mChildHelper.b(view, i, view.getLayoutParams(), false);
            }
            if (pVar.f427d) {
                childViewHolderInt.itemView.invalidate();
                pVar.f427d = false;
            }
        }

        public static int chooseSize(int i, int i2, int i3) {
            int mode = View.MeasureSpec.getMode(i);
            int size = View.MeasureSpec.getSize(i);
            if (mode != Integer.MIN_VALUE) {
                return mode != 1073741824 ? Math.max(i2, i3) : size;
            }
            return Math.min(size, Math.max(i2, i3));
        }

        private void detachViewInternal(int i, View view) {
            this.mChildHelper.c(i);
        }

        /* JADX WARN: Code restructure failed: missing block: B:4:0x000a, code lost:
            if (r3 >= 0) goto L8;
         */
        @Deprecated
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public static int getChildMeasureSpec(int i, int i2, int i3, boolean z) {
            int i4 = i - i2;
            int i5 = 0;
            int max = Math.max(0, i4);
            if (!z) {
                if (i3 < 0) {
                    if (i3 == -1) {
                        i3 = max;
                    } else {
                        if (i3 == -2) {
                            i5 = Integer.MIN_VALUE;
                            i3 = max;
                        }
                        i3 = 0;
                    }
                }
                i5 = 1073741824;
            }
            return View.MeasureSpec.makeMeasureSpec(i3, i5);
        }

        private int[] getChildRectangleOnScreenScrollAmount(View view, Rect rect) {
            int[] iArr = new int[2];
            int paddingLeft = getPaddingLeft();
            int paddingTop = getPaddingTop();
            int width = getWidth() - getPaddingRight();
            int height = getHeight() - getPaddingBottom();
            int left = (view.getLeft() + rect.left) - view.getScrollX();
            int top = (view.getTop() + rect.top) - view.getScrollY();
            int width2 = rect.width() + left;
            int height2 = rect.height() + top;
            int i = left - paddingLeft;
            int min = Math.min(0, i);
            int i2 = top - paddingTop;
            int min2 = Math.min(0, i2);
            int i3 = width2 - width;
            int max = Math.max(0, i3);
            int max2 = Math.max(0, height2 - height);
            if (getLayoutDirection() != 1) {
                if (min == 0) {
                    min = Math.min(i, max);
                }
                max = min;
            } else if (max == 0) {
                max = Math.max(min, i3);
            }
            if (min2 == 0) {
                min2 = Math.min(i2, max2);
            }
            iArr[0] = max;
            iArr[1] = min2;
            return iArr;
        }

        public static d getProperties(Context context, AttributeSet attributeSet, int i, int i2) {
            d dVar = new d();
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.w.a.f2701a, i, i2);
            dVar.f420a = obtainStyledAttributes.getInt(0, 1);
            dVar.f421b = obtainStyledAttributes.getInt(10, 1);
            dVar.f422c = obtainStyledAttributes.getBoolean(9, false);
            dVar.f423d = obtainStyledAttributes.getBoolean(11, false);
            obtainStyledAttributes.recycle();
            return dVar;
        }

        private boolean isFocusedChildVisibleAfterScrolling(RecyclerView recyclerView, int i, int i2) {
            View focusedChild = recyclerView.getFocusedChild();
            if (focusedChild == null) {
                return false;
            }
            int paddingLeft = getPaddingLeft();
            int paddingTop = getPaddingTop();
            int width = getWidth() - getPaddingRight();
            int height = getHeight() - getPaddingBottom();
            Rect rect = this.mRecyclerView.mTempRect;
            getDecoratedBoundsWithMargins(focusedChild, rect);
            return rect.left - i < width && rect.right - i > paddingLeft && rect.top - i2 < height && rect.bottom - i2 > paddingTop;
        }

        private static boolean isMeasurementUpToDate(int i, int i2, int i3) {
            int mode = View.MeasureSpec.getMode(i2);
            int size = View.MeasureSpec.getSize(i2);
            if (i3 <= 0 || i == i3) {
                if (mode == Integer.MIN_VALUE) {
                    return size >= i;
                } else if (mode != 0) {
                    return mode == 1073741824 && size == i;
                } else {
                    return true;
                }
            }
            return false;
        }

        private void scrapOrRecycleView(v vVar, int i, View view) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (childViewHolderInt.shouldIgnore()) {
                return;
            }
            if (childViewHolderInt.isInvalid() && !childViewHolderInt.isRemoved() && !this.mRecyclerView.mAdapter.hasStableIds()) {
                removeViewAt(i);
                vVar.i(childViewHolderInt);
                return;
            }
            detachViewAt(i);
            vVar.j(view);
            this.mRecyclerView.mViewInfoStore.f(childViewHolderInt);
        }

        public void addDisappearingView(View view) {
            addDisappearingView(view, -1);
        }

        public void addView(View view) {
            addView(view, -1);
        }

        public void assertInLayoutOrScroll(String str) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                recyclerView.assertInLayoutOrScroll(str);
            }
        }

        public void assertNotInLayoutOrScroll(String str) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                recyclerView.assertNotInLayoutOrScroll(str);
            }
        }

        public void attachView(View view, int i, p pVar) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (childViewHolderInt.isRemoved()) {
                this.mRecyclerView.mViewInfoStore.a(childViewHolderInt);
            } else {
                this.mRecyclerView.mViewInfoStore.f(childViewHolderInt);
            }
            this.mChildHelper.b(view, i, pVar, childViewHolderInt.isRemoved());
        }

        public void calculateItemDecorationsForChild(View view, Rect rect) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null) {
                rect.set(0, 0, 0, 0);
            } else {
                rect.set(recyclerView.getItemDecorInsetsForChild(view));
            }
        }

        public boolean canScrollHorizontally() {
            return false;
        }

        public boolean canScrollVertically() {
            return false;
        }

        public boolean checkLayoutParams(p pVar) {
            return pVar != null;
        }

        public void collectAdjacentPrefetchPositions(int i, int i2, a0 a0Var, c cVar) {
        }

        public void collectInitialPrefetchPositions(int i, c cVar) {
        }

        public int computeHorizontalScrollExtent(a0 a0Var) {
            return 0;
        }

        public int computeHorizontalScrollOffset(a0 a0Var) {
            return 0;
        }

        public int computeHorizontalScrollRange(a0 a0Var) {
            return 0;
        }

        public int computeVerticalScrollExtent(a0 a0Var) {
            return 0;
        }

        public int computeVerticalScrollOffset(a0 a0Var) {
            return 0;
        }

        public int computeVerticalScrollRange(a0 a0Var) {
            return 0;
        }

        public void detachAndScrapAttachedViews(v vVar) {
            for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
                scrapOrRecycleView(vVar, childCount, getChildAt(childCount));
            }
        }

        public void detachAndScrapView(View view, v vVar) {
            scrapOrRecycleView(vVar, this.mChildHelper.j(view), view);
        }

        public void detachAndScrapViewAt(int i, v vVar) {
            scrapOrRecycleView(vVar, i, getChildAt(i));
        }

        public void detachView(View view) {
            int j = this.mChildHelper.j(view);
            if (j >= 0) {
                detachViewInternal(j, view);
            }
        }

        public void detachViewAt(int i) {
            detachViewInternal(i, getChildAt(i));
        }

        public void dispatchAttachedToWindow(RecyclerView recyclerView) {
            this.mIsAttachedToWindow = true;
            onAttachedToWindow(recyclerView);
        }

        public void dispatchDetachedFromWindow(RecyclerView recyclerView, v vVar) {
            this.mIsAttachedToWindow = false;
            onDetachedFromWindow(recyclerView, vVar);
        }

        public void endAnimation(View view) {
            l lVar = this.mRecyclerView.mItemAnimator;
            if (lVar != null) {
                lVar.e(RecyclerView.getChildViewHolderInt(view));
            }
        }

        public View findContainingItemView(View view) {
            View findContainingItemView;
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null || (findContainingItemView = recyclerView.findContainingItemView(view)) == null || this.mChildHelper.f2714c.contains(findContainingItemView)) {
                return null;
            }
            return findContainingItemView;
        }

        public View findViewByPosition(int i) {
            int childCount = getChildCount();
            for (int i2 = 0; i2 < childCount; i2++) {
                View childAt = getChildAt(i2);
                d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(childAt);
                if (childViewHolderInt != null && childViewHolderInt.getLayoutPosition() == i && !childViewHolderInt.shouldIgnore() && (this.mRecyclerView.mState.f396g || !childViewHolderInt.isRemoved())) {
                    return childAt;
                }
            }
            return null;
        }

        public abstract p generateDefaultLayoutParams();

        public p generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
            if (layoutParams instanceof p) {
                return new p((p) layoutParams);
            }
            if (layoutParams instanceof ViewGroup.MarginLayoutParams) {
                return new p((ViewGroup.MarginLayoutParams) layoutParams);
            }
            return new p(layoutParams);
        }

        public int getBaseline() {
            return -1;
        }

        public int getBottomDecorationHeight(View view) {
            return ((p) view.getLayoutParams()).f425b.bottom;
        }

        public View getChildAt(int i) {
            b.w.b.b bVar = this.mChildHelper;
            if (bVar != null) {
                return ((e) bVar.f2712a).a(bVar.f(i));
            }
            return null;
        }

        public int getChildCount() {
            b.w.b.b bVar = this.mChildHelper;
            if (bVar != null) {
                return bVar.e();
            }
            return 0;
        }

        public boolean getClipToPadding() {
            RecyclerView recyclerView = this.mRecyclerView;
            return recyclerView != null && recyclerView.mClipToPadding;
        }

        public int getColumnCountForAccessibility(v vVar, a0 a0Var) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null || recyclerView.mAdapter == null || !canScrollHorizontally()) {
                return 1;
            }
            return this.mRecyclerView.mAdapter.getItemCount();
        }

        public int getDecoratedBottom(View view) {
            return getBottomDecorationHeight(view) + view.getBottom();
        }

        public void getDecoratedBoundsWithMargins(View view, Rect rect) {
            RecyclerView.getDecoratedBoundsWithMarginsInt(view, rect);
        }

        public int getDecoratedLeft(View view) {
            return view.getLeft() - getLeftDecorationWidth(view);
        }

        public int getDecoratedMeasuredHeight(View view) {
            Rect rect = ((p) view.getLayoutParams()).f425b;
            return view.getMeasuredHeight() + rect.top + rect.bottom;
        }

        public int getDecoratedMeasuredWidth(View view) {
            Rect rect = ((p) view.getLayoutParams()).f425b;
            return view.getMeasuredWidth() + rect.left + rect.right;
        }

        public int getDecoratedRight(View view) {
            return getRightDecorationWidth(view) + view.getRight();
        }

        public int getDecoratedTop(View view) {
            return view.getTop() - getTopDecorationHeight(view);
        }

        public View getFocusedChild() {
            View focusedChild;
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null || (focusedChild = recyclerView.getFocusedChild()) == null || this.mChildHelper.f2714c.contains(focusedChild)) {
                return null;
            }
            return focusedChild;
        }

        public int getHeight() {
            return this.mHeight;
        }

        public int getHeightMode() {
            return this.mHeightMode;
        }

        public int getItemCount() {
            RecyclerView recyclerView = this.mRecyclerView;
            g adapter = recyclerView != null ? recyclerView.getAdapter() : null;
            if (adapter != null) {
                return adapter.getItemCount();
            }
            return 0;
        }

        public int getItemViewType(View view) {
            return RecyclerView.getChildViewHolderInt(view).getItemViewType();
        }

        public int getLayoutDirection() {
            RecyclerView recyclerView = this.mRecyclerView;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return recyclerView.getLayoutDirection();
        }

        public int getLeftDecorationWidth(View view) {
            return ((p) view.getLayoutParams()).f425b.left;
        }

        public int getMinimumHeight() {
            RecyclerView recyclerView = this.mRecyclerView;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return recyclerView.getMinimumHeight();
        }

        public int getMinimumWidth() {
            RecyclerView recyclerView = this.mRecyclerView;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return recyclerView.getMinimumWidth();
        }

        public int getPaddingBottom() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                return recyclerView.getPaddingBottom();
            }
            return 0;
        }

        public int getPaddingEnd() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                return recyclerView.getPaddingEnd();
            }
            return 0;
        }

        public int getPaddingLeft() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                return recyclerView.getPaddingLeft();
            }
            return 0;
        }

        public int getPaddingRight() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                return recyclerView.getPaddingRight();
            }
            return 0;
        }

        public int getPaddingStart() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                return recyclerView.getPaddingStart();
            }
            return 0;
        }

        public int getPaddingTop() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                return recyclerView.getPaddingTop();
            }
            return 0;
        }

        public int getPosition(View view) {
            return ((p) view.getLayoutParams()).a();
        }

        public int getRightDecorationWidth(View view) {
            return ((p) view.getLayoutParams()).f425b.right;
        }

        public int getRowCountForAccessibility(v vVar, a0 a0Var) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null || recyclerView.mAdapter == null || !canScrollVertically()) {
                return 1;
            }
            return this.mRecyclerView.mAdapter.getItemCount();
        }

        public int getSelectionModeForAccessibility(v vVar, a0 a0Var) {
            return 0;
        }

        public int getTopDecorationHeight(View view) {
            return ((p) view.getLayoutParams()).f425b.top;
        }

        public void getTransformedBoundingBox(View view, boolean z, Rect rect) {
            Matrix matrix;
            if (z) {
                Rect rect2 = ((p) view.getLayoutParams()).f425b;
                rect.set(-rect2.left, -rect2.top, view.getWidth() + rect2.right, view.getHeight() + rect2.bottom);
            } else {
                rect.set(0, 0, view.getWidth(), view.getHeight());
            }
            if (this.mRecyclerView != null && (matrix = view.getMatrix()) != null && !matrix.isIdentity()) {
                RectF rectF = this.mRecyclerView.mTempRectF;
                rectF.set(rect);
                matrix.mapRect(rectF);
                rect.set((int) Math.floor(rectF.left), (int) Math.floor(rectF.top), (int) Math.ceil(rectF.right), (int) Math.ceil(rectF.bottom));
            }
            rect.offset(view.getLeft(), view.getTop());
        }

        public int getWidth() {
            return this.mWidth;
        }

        public int getWidthMode() {
            return this.mWidthMode;
        }

        public boolean hasFlexibleChildInBothOrientations() {
            int childCount = getChildCount();
            for (int i = 0; i < childCount; i++) {
                ViewGroup.LayoutParams layoutParams = getChildAt(i).getLayoutParams();
                if (layoutParams.width < 0 && layoutParams.height < 0) {
                    return true;
                }
            }
            return false;
        }

        public boolean hasFocus() {
            RecyclerView recyclerView = this.mRecyclerView;
            return recyclerView != null && recyclerView.hasFocus();
        }

        public void ignoreView(View view) {
            ViewParent parent = view.getParent();
            RecyclerView recyclerView = this.mRecyclerView;
            if (parent == recyclerView && recyclerView.indexOfChild(view) != -1) {
                d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
                childViewHolderInt.addFlags(128);
                this.mRecyclerView.mViewInfoStore.g(childViewHolderInt);
                return;
            }
            throw new IllegalArgumentException(c.b.a.a.a.i(this.mRecyclerView, c.b.a.a.a.x("View should be fully attached to be ignored")));
        }

        public boolean isAttachedToWindow() {
            return this.mIsAttachedToWindow;
        }

        public boolean isAutoMeasureEnabled() {
            return this.mAutoMeasure;
        }

        public boolean isFocused() {
            RecyclerView recyclerView = this.mRecyclerView;
            return recyclerView != null && recyclerView.isFocused();
        }

        public final boolean isItemPrefetchEnabled() {
            return this.mItemPrefetchEnabled;
        }

        public boolean isLayoutHierarchical(v vVar, a0 a0Var) {
            return false;
        }

        public boolean isMeasurementCacheEnabled() {
            return this.mMeasurementCacheEnabled;
        }

        public boolean isSmoothScrolling() {
            z zVar = this.mSmoothScroller;
            return zVar != null && zVar.isRunning();
        }

        public boolean isViewPartiallyVisible(View view, boolean z, boolean z2) {
            boolean z3 = this.mHorizontalBoundCheck.b(view, 24579) && this.mVerticalBoundCheck.b(view, 24579);
            return z ? z3 : !z3;
        }

        public void layoutDecorated(View view, int i, int i2, int i3, int i4) {
            Rect rect = ((p) view.getLayoutParams()).f425b;
            view.layout(i + rect.left, i2 + rect.top, i3 - rect.right, i4 - rect.bottom);
        }

        public void layoutDecoratedWithMargins(View view, int i, int i2, int i3, int i4) {
            p pVar = (p) view.getLayoutParams();
            Rect rect = pVar.f425b;
            view.layout(i + rect.left + ((ViewGroup.MarginLayoutParams) pVar).leftMargin, i2 + rect.top + ((ViewGroup.MarginLayoutParams) pVar).topMargin, (i3 - rect.right) - ((ViewGroup.MarginLayoutParams) pVar).rightMargin, (i4 - rect.bottom) - ((ViewGroup.MarginLayoutParams) pVar).bottomMargin);
        }

        public void measureChild(View view, int i, int i2) {
            p pVar = (p) view.getLayoutParams();
            Rect itemDecorInsetsForChild = this.mRecyclerView.getItemDecorInsetsForChild(view);
            int i3 = itemDecorInsetsForChild.left + itemDecorInsetsForChild.right + i;
            int i4 = itemDecorInsetsForChild.top + itemDecorInsetsForChild.bottom + i2;
            int childMeasureSpec = getChildMeasureSpec(getWidth(), getWidthMode(), getPaddingRight() + getPaddingLeft() + i3, ((ViewGroup.MarginLayoutParams) pVar).width, canScrollHorizontally());
            int childMeasureSpec2 = getChildMeasureSpec(getHeight(), getHeightMode(), getPaddingBottom() + getPaddingTop() + i4, ((ViewGroup.MarginLayoutParams) pVar).height, canScrollVertically());
            if (shouldMeasureChild(view, childMeasureSpec, childMeasureSpec2, pVar)) {
                view.measure(childMeasureSpec, childMeasureSpec2);
            }
        }

        public void measureChildWithMargins(View view, int i, int i2) {
            p pVar = (p) view.getLayoutParams();
            Rect itemDecorInsetsForChild = this.mRecyclerView.getItemDecorInsetsForChild(view);
            int i3 = itemDecorInsetsForChild.left + itemDecorInsetsForChild.right + i;
            int i4 = itemDecorInsetsForChild.top + itemDecorInsetsForChild.bottom + i2;
            int childMeasureSpec = getChildMeasureSpec(getWidth(), getWidthMode(), getPaddingRight() + getPaddingLeft() + ((ViewGroup.MarginLayoutParams) pVar).leftMargin + ((ViewGroup.MarginLayoutParams) pVar).rightMargin + i3, ((ViewGroup.MarginLayoutParams) pVar).width, canScrollHorizontally());
            int childMeasureSpec2 = getChildMeasureSpec(getHeight(), getHeightMode(), getPaddingBottom() + getPaddingTop() + ((ViewGroup.MarginLayoutParams) pVar).topMargin + ((ViewGroup.MarginLayoutParams) pVar).bottomMargin + i4, ((ViewGroup.MarginLayoutParams) pVar).height, canScrollVertically());
            if (shouldMeasureChild(view, childMeasureSpec, childMeasureSpec2, pVar)) {
                view.measure(childMeasureSpec, childMeasureSpec2);
            }
        }

        public void moveView(int i, int i2) {
            View childAt = getChildAt(i);
            if (childAt != null) {
                detachViewAt(i);
                attachView(childAt, i2);
                return;
            }
            throw new IllegalArgumentException("Cannot move a child from non-existing index:" + i + this.mRecyclerView.toString());
        }

        public void offsetChildrenHorizontal(int i) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                recyclerView.offsetChildrenHorizontal(i);
            }
        }

        public void offsetChildrenVertical(int i) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                recyclerView.offsetChildrenVertical(i);
            }
        }

        public void onAdapterChanged(g gVar, g gVar2) {
        }

        public boolean onAddFocusables(RecyclerView recyclerView, ArrayList<View> arrayList, int i, int i2) {
            return false;
        }

        public void onAttachedToWindow(RecyclerView recyclerView) {
        }

        @Deprecated
        public void onDetachedFromWindow(RecyclerView recyclerView) {
        }

        public void onDetachedFromWindow(RecyclerView recyclerView, v vVar) {
            onDetachedFromWindow(recyclerView);
        }

        public View onFocusSearchFailed(View view, int i, v vVar, a0 a0Var) {
            return null;
        }

        public void onInitializeAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
            RecyclerView recyclerView = this.mRecyclerView;
            onInitializeAccessibilityEvent(recyclerView.mRecycler, recyclerView.mState, accessibilityEvent);
        }

        public void onInitializeAccessibilityNodeInfo(b.j.j.x.b bVar) {
            RecyclerView recyclerView = this.mRecyclerView;
            onInitializeAccessibilityNodeInfo(recyclerView.mRecycler, recyclerView.mState, bVar);
        }

        public void onInitializeAccessibilityNodeInfoForItem(View view, b.j.j.x.b bVar) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (childViewHolderInt == null || childViewHolderInt.isRemoved() || this.mChildHelper.k(childViewHolderInt.itemView)) {
                return;
            }
            RecyclerView recyclerView = this.mRecyclerView;
            onInitializeAccessibilityNodeInfoForItem(recyclerView.mRecycler, recyclerView.mState, view, bVar);
        }

        public View onInterceptFocusSearch(View view, int i) {
            return null;
        }

        public void onItemsAdded(RecyclerView recyclerView, int i, int i2) {
        }

        public void onItemsChanged(RecyclerView recyclerView) {
        }

        public void onItemsMoved(RecyclerView recyclerView, int i, int i2, int i3) {
        }

        public void onItemsRemoved(RecyclerView recyclerView, int i, int i2) {
        }

        public void onItemsUpdated(RecyclerView recyclerView, int i, int i2) {
        }

        public void onItemsUpdated(RecyclerView recyclerView, int i, int i2, Object obj) {
            onItemsUpdated(recyclerView, i, i2);
        }

        public void onLayoutChildren(v vVar, a0 a0Var) {
            Log.e(RecyclerView.TAG, "You must override onLayoutChildren(Recycler recycler, State state) ");
        }

        public void onLayoutCompleted(a0 a0Var) {
        }

        public void onMeasure(v vVar, a0 a0Var, int i, int i2) {
            this.mRecyclerView.defaultOnMeasure(i, i2);
        }

        @Deprecated
        public boolean onRequestChildFocus(RecyclerView recyclerView, View view, View view2) {
            return isSmoothScrolling() || recyclerView.isComputingLayout();
        }

        public void onRestoreInstanceState(Parcelable parcelable) {
        }

        public Parcelable onSaveInstanceState() {
            return null;
        }

        public void onScrollStateChanged(int i) {
        }

        public void onSmoothScrollerStopped(z zVar) {
            if (this.mSmoothScroller == zVar) {
                this.mSmoothScroller = null;
            }
        }

        public boolean performAccessibilityAction(int i, Bundle bundle) {
            RecyclerView recyclerView = this.mRecyclerView;
            return performAccessibilityAction(recyclerView.mRecycler, recyclerView.mState, i, bundle);
        }

        public boolean performAccessibilityActionForItem(View view, int i, Bundle bundle) {
            RecyclerView recyclerView = this.mRecyclerView;
            return performAccessibilityActionForItem(recyclerView.mRecycler, recyclerView.mState, view, i, bundle);
        }

        public boolean performAccessibilityActionForItem(v vVar, a0 a0Var, View view, int i, Bundle bundle) {
            return false;
        }

        public void postOnAnimation(Runnable runnable) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                recyclerView.postOnAnimation(runnable);
            }
        }

        public void removeAllViews() {
            for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
                this.mChildHelper.l(childCount);
            }
        }

        public void removeAndRecycleAllViews(v vVar) {
            for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
                if (!RecyclerView.getChildViewHolderInt(getChildAt(childCount)).shouldIgnore()) {
                    removeAndRecycleViewAt(childCount, vVar);
                }
            }
        }

        public void removeAndRecycleScrapInt(v vVar) {
            int size = vVar.f434a.size();
            for (int i = size - 1; i >= 0; i--) {
                View view = vVar.f434a.get(i).itemView;
                d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
                if (!childViewHolderInt.shouldIgnore()) {
                    childViewHolderInt.setIsRecyclable(false);
                    if (childViewHolderInt.isTmpDetached()) {
                        this.mRecyclerView.removeDetachedView(view, false);
                    }
                    l lVar = this.mRecyclerView.mItemAnimator;
                    if (lVar != null) {
                        lVar.e(childViewHolderInt);
                    }
                    childViewHolderInt.setIsRecyclable(true);
                    d0 childViewHolderInt2 = RecyclerView.getChildViewHolderInt(view);
                    childViewHolderInt2.mScrapContainer = null;
                    childViewHolderInt2.mInChangeScrap = false;
                    childViewHolderInt2.clearReturnedFromScrapFlag();
                    vVar.i(childViewHolderInt2);
                }
            }
            vVar.f434a.clear();
            ArrayList<d0> arrayList = vVar.f435b;
            if (arrayList != null) {
                arrayList.clear();
            }
            if (size > 0) {
                this.mRecyclerView.invalidate();
            }
        }

        public void removeAndRecycleView(View view, v vVar) {
            removeView(view);
            vVar.h(view);
        }

        public void removeAndRecycleViewAt(int i, v vVar) {
            View childAt = getChildAt(i);
            removeViewAt(i);
            vVar.h(childAt);
        }

        public boolean removeCallbacks(Runnable runnable) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                return recyclerView.removeCallbacks(runnable);
            }
            return false;
        }

        public void removeDetachedView(View view) {
            this.mRecyclerView.removeDetachedView(view, false);
        }

        public void removeView(View view) {
            b.w.b.b bVar = this.mChildHelper;
            int indexOfChild = RecyclerView.this.indexOfChild(view);
            if (indexOfChild < 0) {
                return;
            }
            if (bVar.f2713b.f(indexOfChild)) {
                bVar.m(view);
            }
            ((e) bVar.f2712a).c(indexOfChild);
        }

        public void removeViewAt(int i) {
            if (getChildAt(i) != null) {
                this.mChildHelper.l(i);
            }
        }

        public boolean requestChildRectangleOnScreen(RecyclerView recyclerView, View view, Rect rect, boolean z) {
            return requestChildRectangleOnScreen(recyclerView, view, rect, z, false);
        }

        public void requestLayout() {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView != null) {
                recyclerView.requestLayout();
            }
        }

        public void requestSimpleAnimationsInNextLayout() {
            this.mRequestedSimpleAnimations = true;
        }

        public int scrollHorizontallyBy(int i, v vVar, a0 a0Var) {
            return 0;
        }

        public void scrollToPosition(int i) {
        }

        public int scrollVerticallyBy(int i, v vVar, a0 a0Var) {
            return 0;
        }

        @Deprecated
        public void setAutoMeasureEnabled(boolean z) {
            this.mAutoMeasure = z;
        }

        public void setExactMeasureSpecsFrom(RecyclerView recyclerView) {
            setMeasureSpecs(View.MeasureSpec.makeMeasureSpec(recyclerView.getWidth(), 1073741824), View.MeasureSpec.makeMeasureSpec(recyclerView.getHeight(), 1073741824));
        }

        public final void setItemPrefetchEnabled(boolean z) {
            if (z != this.mItemPrefetchEnabled) {
                this.mItemPrefetchEnabled = z;
                this.mPrefetchMaxCountObserved = 0;
                RecyclerView recyclerView = this.mRecyclerView;
                if (recyclerView != null) {
                    recyclerView.mRecycler.m();
                }
            }
        }

        public void setMeasureSpecs(int i, int i2) {
            this.mWidth = View.MeasureSpec.getSize(i);
            int mode = View.MeasureSpec.getMode(i);
            this.mWidthMode = mode;
            if (mode == 0 && !RecyclerView.ALLOW_SIZE_IN_UNSPECIFIED_SPEC) {
                this.mWidth = 0;
            }
            this.mHeight = View.MeasureSpec.getSize(i2);
            int mode2 = View.MeasureSpec.getMode(i2);
            this.mHeightMode = mode2;
            if (mode2 != 0 || RecyclerView.ALLOW_SIZE_IN_UNSPECIFIED_SPEC) {
                return;
            }
            this.mHeight = 0;
        }

        public void setMeasuredDimension(Rect rect, int i, int i2) {
            setMeasuredDimension(chooseSize(i, getPaddingRight() + getPaddingLeft() + rect.width(), getMinimumWidth()), chooseSize(i2, getPaddingBottom() + getPaddingTop() + rect.height(), getMinimumHeight()));
        }

        public void setMeasuredDimensionFromChildren(int i, int i2) {
            int childCount = getChildCount();
            if (childCount == 0) {
                this.mRecyclerView.defaultOnMeasure(i, i2);
                return;
            }
            int i3 = Integer.MIN_VALUE;
            int i4 = Integer.MAX_VALUE;
            int i5 = Integer.MAX_VALUE;
            int i6 = Integer.MIN_VALUE;
            for (int i7 = 0; i7 < childCount; i7++) {
                View childAt = getChildAt(i7);
                Rect rect = this.mRecyclerView.mTempRect;
                getDecoratedBoundsWithMargins(childAt, rect);
                int i8 = rect.left;
                if (i8 < i4) {
                    i4 = i8;
                }
                int i9 = rect.right;
                if (i9 > i3) {
                    i3 = i9;
                }
                int i10 = rect.top;
                if (i10 < i5) {
                    i5 = i10;
                }
                int i11 = rect.bottom;
                if (i11 > i6) {
                    i6 = i11;
                }
            }
            this.mRecyclerView.mTempRect.set(i4, i5, i3, i6);
            setMeasuredDimension(this.mRecyclerView.mTempRect, i, i2);
        }

        public void setMeasurementCacheEnabled(boolean z) {
            this.mMeasurementCacheEnabled = z;
        }

        public void setRecyclerView(RecyclerView recyclerView) {
            if (recyclerView == null) {
                this.mRecyclerView = null;
                this.mChildHelper = null;
                this.mWidth = 0;
                this.mHeight = 0;
            } else {
                this.mRecyclerView = recyclerView;
                this.mChildHelper = recyclerView.mChildHelper;
                this.mWidth = recyclerView.getWidth();
                this.mHeight = recyclerView.getHeight();
            }
            this.mWidthMode = 1073741824;
            this.mHeightMode = 1073741824;
        }

        public boolean shouldMeasureChild(View view, int i, int i2, p pVar) {
            return (!view.isLayoutRequested() && this.mMeasurementCacheEnabled && isMeasurementUpToDate(view.getWidth(), i, ((ViewGroup.MarginLayoutParams) pVar).width) && isMeasurementUpToDate(view.getHeight(), i2, ((ViewGroup.MarginLayoutParams) pVar).height)) ? false : true;
        }

        public boolean shouldMeasureTwice() {
            return false;
        }

        public boolean shouldReMeasureChild(View view, int i, int i2, p pVar) {
            return (this.mMeasurementCacheEnabled && isMeasurementUpToDate(view.getMeasuredWidth(), i, ((ViewGroup.MarginLayoutParams) pVar).width) && isMeasurementUpToDate(view.getMeasuredHeight(), i2, ((ViewGroup.MarginLayoutParams) pVar).height)) ? false : true;
        }

        public void smoothScrollToPosition(RecyclerView recyclerView, a0 a0Var, int i) {
            Log.e(RecyclerView.TAG, "You must override smoothScrollToPosition to support smooth scrolling");
        }

        public void startSmoothScroll(z zVar) {
            z zVar2 = this.mSmoothScroller;
            if (zVar2 != null && zVar != zVar2 && zVar2.isRunning()) {
                this.mSmoothScroller.stop();
            }
            this.mSmoothScroller = zVar;
            zVar.start(this.mRecyclerView, this);
        }

        public void stopIgnoringView(View view) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            childViewHolderInt.stopIgnoring();
            childViewHolderInt.resetInternal();
            childViewHolderInt.addFlags(4);
        }

        public void stopSmoothScroller() {
            z zVar = this.mSmoothScroller;
            if (zVar != null) {
                zVar.stop();
            }
        }

        public boolean supportsPredictiveItemAnimations() {
            return false;
        }

        public void addDisappearingView(View view, int i) {
            addViewInt(view, i, true);
        }

        public void addView(View view, int i) {
            addViewInt(view, i, false);
        }

        public void onInitializeAccessibilityEvent(v vVar, a0 a0Var, AccessibilityEvent accessibilityEvent) {
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null || accessibilityEvent == null) {
                return;
            }
            boolean z = true;
            if (!recyclerView.canScrollVertically(1) && !this.mRecyclerView.canScrollVertically(-1) && !this.mRecyclerView.canScrollHorizontally(-1) && !this.mRecyclerView.canScrollHorizontally(1)) {
                z = false;
            }
            accessibilityEvent.setScrollable(z);
            g gVar = this.mRecyclerView.mAdapter;
            if (gVar != null) {
                accessibilityEvent.setItemCount(gVar.getItemCount());
            }
        }

        public void onInitializeAccessibilityNodeInfo(v vVar, a0 a0Var, b.j.j.x.b bVar) {
            if (this.mRecyclerView.canScrollVertically(-1) || this.mRecyclerView.canScrollHorizontally(-1)) {
                bVar.f2259b.addAction(8192);
                bVar.f2259b.setScrollable(true);
            }
            if (this.mRecyclerView.canScrollVertically(1) || this.mRecyclerView.canScrollHorizontally(1)) {
                bVar.f2259b.addAction(4096);
                bVar.f2259b.setScrollable(true);
            }
            bVar.m(b.C0037b.a(getRowCountForAccessibility(vVar, a0Var), getColumnCountForAccessibility(vVar, a0Var), isLayoutHierarchical(vVar, a0Var), getSelectionModeForAccessibility(vVar, a0Var)));
        }

        public boolean onRequestChildFocus(RecyclerView recyclerView, a0 a0Var, View view, View view2) {
            return onRequestChildFocus(recyclerView, view, view2);
        }

        public boolean performAccessibilityAction(v vVar, a0 a0Var, int i, Bundle bundle) {
            int height;
            int width;
            int i2;
            int i3;
            RecyclerView recyclerView = this.mRecyclerView;
            if (recyclerView == null) {
                return false;
            }
            if (i == 4096) {
                height = recyclerView.canScrollVertically(1) ? (getHeight() - getPaddingTop()) - getPaddingBottom() : 0;
                if (this.mRecyclerView.canScrollHorizontally(1)) {
                    width = (getWidth() - getPaddingLeft()) - getPaddingRight();
                    i2 = height;
                    i3 = width;
                }
                i2 = height;
                i3 = 0;
            } else if (i != 8192) {
                i3 = 0;
                i2 = 0;
            } else {
                height = recyclerView.canScrollVertically(-1) ? -((getHeight() - getPaddingTop()) - getPaddingBottom()) : 0;
                if (this.mRecyclerView.canScrollHorizontally(-1)) {
                    width = -((getWidth() - getPaddingLeft()) - getPaddingRight());
                    i2 = height;
                    i3 = width;
                }
                i2 = height;
                i3 = 0;
            }
            if (i2 == 0 && i3 == 0) {
                return false;
            }
            this.mRecyclerView.smoothScrollBy(i3, i2, null, Integer.MIN_VALUE, true);
            return true;
        }

        public boolean requestChildRectangleOnScreen(RecyclerView recyclerView, View view, Rect rect, boolean z, boolean z2) {
            int[] childRectangleOnScreenScrollAmount = getChildRectangleOnScreenScrollAmount(view, rect);
            int i = childRectangleOnScreenScrollAmount[0];
            int i2 = childRectangleOnScreenScrollAmount[1];
            if ((!z2 || isFocusedChildVisibleAfterScrolling(recyclerView, i, i2)) && !(i == 0 && i2 == 0)) {
                if (z) {
                    recyclerView.scrollBy(i, i2);
                } else {
                    recyclerView.smoothScrollBy(i, i2);
                }
                return true;
            }
            return false;
        }

        /* JADX WARN: Code restructure failed: missing block: B:9:0x0017, code lost:
            if (r5 == 1073741824) goto L8;
         */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public static int getChildMeasureSpec(int i, int i2, int i3, int i4, boolean z) {
            int max = Math.max(0, i - i3);
            if (z) {
                if (i4 < 0) {
                    if (i4 == -1) {
                        if (i2 != Integer.MIN_VALUE) {
                            if (i2 != 0) {
                            }
                        }
                        i4 = max;
                    }
                    i2 = 0;
                    i4 = 0;
                }
                i2 = 1073741824;
            } else {
                if (i4 < 0) {
                    if (i4 != -1) {
                        if (i4 == -2) {
                            if (i2 == Integer.MIN_VALUE || i2 == 1073741824) {
                                i4 = max;
                                i2 = Integer.MIN_VALUE;
                            } else {
                                i4 = max;
                                i2 = 0;
                            }
                        }
                        i2 = 0;
                        i4 = 0;
                    }
                    i4 = max;
                }
                i2 = 1073741824;
            }
            return View.MeasureSpec.makeMeasureSpec(i4, i2);
        }

        public void onInitializeAccessibilityNodeInfoForItem(v vVar, a0 a0Var, View view, b.j.j.x.b bVar) {
            bVar.n(b.c.a(canScrollVertically() ? getPosition(view) : 0, 1, canScrollHorizontally() ? getPosition(view) : 0, 1, false, false));
        }

        public void attachView(View view, int i) {
            attachView(view, i, (p) view.getLayoutParams());
        }

        public p generateLayoutParams(Context context, AttributeSet attributeSet) {
            return new p(context, attributeSet);
        }

        public void setMeasuredDimension(int i, int i2) {
            this.mRecyclerView.setMeasuredDimension(i, i2);
        }

        public void attachView(View view) {
            attachView(view, -1);
        }
    }

    /* loaded from: classes.dex */
    public interface q {
        void a(View view);

        void b(View view);
    }

    /* loaded from: classes.dex */
    public static abstract class r {
    }

    /* loaded from: classes.dex */
    public interface s {
        void a(RecyclerView recyclerView, MotionEvent motionEvent);

        boolean b(RecyclerView recyclerView, MotionEvent motionEvent);

        void c(boolean z);
    }

    /* loaded from: classes.dex */
    public static abstract class t {
        public void onScrollStateChanged(RecyclerView recyclerView, int i) {
        }

        public void onScrolled(RecyclerView recyclerView, int i, int i2) {
        }
    }

    /* loaded from: classes.dex */
    public static class u {

        /* renamed from: a  reason: collision with root package name */
        public SparseArray<a> f428a = new SparseArray<>();

        /* renamed from: b  reason: collision with root package name */
        public int f429b = 0;

        /* loaded from: classes.dex */
        public static class a {

            /* renamed from: a  reason: collision with root package name */
            public final ArrayList<d0> f430a = new ArrayList<>();

            /* renamed from: b  reason: collision with root package name */
            public int f431b = 5;

            /* renamed from: c  reason: collision with root package name */
            public long f432c = 0;

            /* renamed from: d  reason: collision with root package name */
            public long f433d = 0;
        }

        public final a a(int i) {
            a aVar = this.f428a.get(i);
            if (aVar == null) {
                a aVar2 = new a();
                this.f428a.put(i, aVar2);
                return aVar2;
            }
            return aVar;
        }

        public long b(long j, long j2) {
            if (j == 0) {
                return j2;
            }
            return (j2 / 4) + ((j / 4) * 3);
        }
    }

    /* loaded from: classes.dex */
    public final class v {

        /* renamed from: a  reason: collision with root package name */
        public final ArrayList<d0> f434a;

        /* renamed from: b  reason: collision with root package name */
        public ArrayList<d0> f435b;

        /* renamed from: c  reason: collision with root package name */
        public final ArrayList<d0> f436c;

        /* renamed from: d  reason: collision with root package name */
        public final List<d0> f437d;

        /* renamed from: e  reason: collision with root package name */
        public int f438e;

        /* renamed from: f  reason: collision with root package name */
        public int f439f;

        /* renamed from: g  reason: collision with root package name */
        public u f440g;

        public v() {
            ArrayList<d0> arrayList = new ArrayList<>();
            this.f434a = arrayList;
            this.f435b = null;
            this.f436c = new ArrayList<>();
            this.f437d = Collections.unmodifiableList(arrayList);
            this.f438e = 2;
            this.f439f = 2;
        }

        public void a(d0 d0Var, boolean z) {
            RecyclerView.clearNestedRecyclerViewIfNotNested(d0Var);
            View view = d0Var.itemView;
            b.w.b.v vVar = RecyclerView.this.mAccessibilityDelegate;
            if (vVar != null) {
                b.j.j.a itemDelegate = vVar.getItemDelegate();
                b.j.j.q.n(view, itemDelegate instanceof v.a ? ((v.a) itemDelegate).f2801b.remove(view) : null);
            }
            if (z) {
                w wVar = RecyclerView.this.mRecyclerListener;
                if (wVar != null) {
                    wVar.a(d0Var);
                }
                g gVar = RecyclerView.this.mAdapter;
                if (gVar != null) {
                    gVar.onViewRecycled(d0Var);
                }
                RecyclerView recyclerView = RecyclerView.this;
                if (recyclerView.mState != null) {
                    recyclerView.mViewInfoStore.g(d0Var);
                }
            }
            d0Var.mOwnerRecyclerView = null;
            u d2 = d();
            Objects.requireNonNull(d2);
            int itemViewType = d0Var.getItemViewType();
            ArrayList<d0> arrayList = d2.a(itemViewType).f430a;
            if (d2.f428a.get(itemViewType).f431b <= arrayList.size()) {
                return;
            }
            d0Var.resetInternal();
            arrayList.add(d0Var);
        }

        public void b() {
            this.f434a.clear();
            f();
        }

        public int c(int i) {
            if (i >= 0 && i < RecyclerView.this.mState.b()) {
                RecyclerView recyclerView = RecyclerView.this;
                return !recyclerView.mState.f396g ? i : recyclerView.mAdapterHelper.f(i, 0);
            }
            StringBuilder y = c.b.a.a.a.y("invalid position ", i, ". State item count is ");
            y.append(RecyclerView.this.mState.b());
            throw new IndexOutOfBoundsException(c.b.a.a.a.i(RecyclerView.this, y));
        }

        public u d() {
            if (this.f440g == null) {
                this.f440g = new u();
            }
            return this.f440g;
        }

        public final void e(ViewGroup viewGroup, boolean z) {
            for (int childCount = viewGroup.getChildCount() - 1; childCount >= 0; childCount--) {
                View childAt = viewGroup.getChildAt(childCount);
                if (childAt instanceof ViewGroup) {
                    e((ViewGroup) childAt, true);
                }
            }
            if (z) {
                if (viewGroup.getVisibility() == 4) {
                    viewGroup.setVisibility(0);
                    viewGroup.setVisibility(4);
                    return;
                }
                int visibility = viewGroup.getVisibility();
                viewGroup.setVisibility(4);
                viewGroup.setVisibility(visibility);
            }
        }

        public void f() {
            for (int size = this.f436c.size() - 1; size >= 0; size--) {
                g(size);
            }
            this.f436c.clear();
            if (RecyclerView.ALLOW_THREAD_GAP_WORK) {
                m.b bVar = RecyclerView.this.mPrefetchRegistry;
                int[] iArr = bVar.f2778c;
                if (iArr != null) {
                    Arrays.fill(iArr, -1);
                }
                bVar.f2779d = 0;
            }
        }

        public void g(int i) {
            a(this.f436c.get(i), true);
            this.f436c.remove(i);
        }

        public void h(View view) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (childViewHolderInt.isTmpDetached()) {
                RecyclerView.this.removeDetachedView(view, false);
            }
            if (childViewHolderInt.isScrap()) {
                childViewHolderInt.unScrap();
            } else if (childViewHolderInt.wasReturnedFromScrap()) {
                childViewHolderInt.clearReturnedFromScrapFlag();
            }
            i(childViewHolderInt);
            if (RecyclerView.this.mItemAnimator == null || childViewHolderInt.isRecyclable()) {
                return;
            }
            RecyclerView.this.mItemAnimator.e(childViewHolderInt);
        }

        public void i(d0 d0Var) {
            boolean z;
            boolean z2;
            if (!d0Var.isScrap() && d0Var.itemView.getParent() == null) {
                if (!d0Var.isTmpDetached()) {
                    if (!d0Var.shouldIgnore()) {
                        boolean doesTransientStatePreventRecycling = d0Var.doesTransientStatePreventRecycling();
                        g gVar = RecyclerView.this.mAdapter;
                        if ((gVar != null && doesTransientStatePreventRecycling && gVar.onFailedToRecycleView(d0Var)) || d0Var.isRecyclable()) {
                            if (this.f439f <= 0 || d0Var.hasAnyOfTheFlags(526)) {
                                z = false;
                            } else {
                                int size = this.f436c.size();
                                if (size >= this.f439f && size > 0) {
                                    g(0);
                                    size--;
                                }
                                if (RecyclerView.ALLOW_THREAD_GAP_WORK && size > 0 && !RecyclerView.this.mPrefetchRegistry.c(d0Var.mPosition)) {
                                    do {
                                        size--;
                                        if (size < 0) {
                                            break;
                                        }
                                    } while (RecyclerView.this.mPrefetchRegistry.c(this.f436c.get(size).mPosition));
                                    size++;
                                }
                                this.f436c.add(size, d0Var);
                                z = true;
                            }
                            if (!z) {
                                a(d0Var, true);
                                r1 = true;
                            }
                            z2 = r1;
                            r1 = z;
                        } else {
                            z2 = false;
                        }
                        RecyclerView.this.mViewInfoStore.g(d0Var);
                        if (r1 || z2 || !doesTransientStatePreventRecycling) {
                            return;
                        }
                        d0Var.mOwnerRecyclerView = null;
                        return;
                    }
                    throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, c.b.a.a.a.x("Trying to recycle an ignored view holder. You should first call stopIgnoringView(view) before calling recycle.")));
                }
                StringBuilder sb = new StringBuilder();
                sb.append("Tmp detached view should be removed from RecyclerView before it can be recycled: ");
                sb.append(d0Var);
                throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, sb));
            }
            StringBuilder x = c.b.a.a.a.x("Scrapped or attached views may not be recycled. isScrap:");
            x.append(d0Var.isScrap());
            x.append(" isAttached:");
            x.append(d0Var.itemView.getParent() != null);
            throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, x));
        }

        public void j(View view) {
            d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
            if (!childViewHolderInt.hasAnyOfTheFlags(12) && childViewHolderInt.isUpdated() && !RecyclerView.this.canReuseUpdatedViewHolder(childViewHolderInt)) {
                if (this.f435b == null) {
                    this.f435b = new ArrayList<>();
                }
                childViewHolderInt.setScrapContainer(this, true);
                this.f435b.add(childViewHolderInt);
            } else if (childViewHolderInt.isInvalid() && !childViewHolderInt.isRemoved() && !RecyclerView.this.mAdapter.hasStableIds()) {
                throw new IllegalArgumentException(c.b.a.a.a.i(RecyclerView.this, c.b.a.a.a.x("Called scrap view with an invalid view. Invalid views cannot be reused from scrap, they should rebound from recycler pool.")));
            } else {
                childViewHolderInt.setScrapContainer(this, false);
                this.f434a.add(childViewHolderInt);
            }
        }

        /* JADX WARN: Code restructure failed: missing block: B:162:0x0318, code lost:
            r7 = r14;
         */
        /* JADX WARN: Code restructure failed: missing block: B:166:0x0323, code lost:
            r7 = null;
         */
        /* JADX WARN: Code restructure failed: missing block: B:236:0x0475, code lost:
            if ((r8 == 0 || r8 + r10 < r20) == false) goto L230;
         */
        /* JADX WARN: Removed duplicated region for block: B:130:0x025d  */
        /* JADX WARN: Removed duplicated region for block: B:215:0x040c  */
        /* JADX WARN: Removed duplicated region for block: B:229:0x045e  */
        /* JADX WARN: Removed duplicated region for block: B:240:0x04a2  */
        /* JADX WARN: Removed duplicated region for block: B:255:0x04d9  */
        /* JADX WARN: Removed duplicated region for block: B:258:0x04e2  */
        /* JADX WARN: Removed duplicated region for block: B:262:0x04ed  */
        /* JADX WARN: Removed duplicated region for block: B:263:0x04fb  */
        /* JADX WARN: Removed duplicated region for block: B:269:0x0517 A[ADDED_TO_REGION] */
        /* JADX WARN: Removed duplicated region for block: B:37:0x008f  */
        /* JADX WARN: Removed duplicated region for block: B:42:0x0096  */
        /* JADX WARN: Removed duplicated region for block: B:96:0x01c2  */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public d0 k(int i, boolean z, long j) {
            d0 d0Var;
            boolean z2;
            boolean z3;
            boolean z4;
            boolean z5;
            ViewGroup.LayoutParams layoutParams;
            p pVar;
            RecyclerView findNestedRecyclerView;
            d0 d0Var2;
            d0 d0Var3;
            d0 d0Var4;
            View view;
            boolean z6;
            int size;
            int f2;
            if (i >= 0 && i < RecyclerView.this.mState.b()) {
                int i2 = 32;
                boolean z7 = false;
                if (RecyclerView.this.mState.f396g) {
                    ArrayList<d0> arrayList = this.f435b;
                    if (arrayList != null && (size = arrayList.size()) != 0) {
                        int i3 = 0;
                        while (true) {
                            if (i3 < size) {
                                d0Var = this.f435b.get(i3);
                                if (!d0Var.wasReturnedFromScrap() && d0Var.getLayoutPosition() == i) {
                                    d0Var.addFlags(32);
                                    break;
                                }
                                i3++;
                            } else if (RecyclerView.this.mAdapter.hasStableIds() && (f2 = RecyclerView.this.mAdapterHelper.f(i, 0)) > 0 && f2 < RecyclerView.this.mAdapter.getItemCount()) {
                                long itemId = RecyclerView.this.mAdapter.getItemId(f2);
                                for (int i4 = 0; i4 < size; i4++) {
                                    d0 d0Var5 = this.f435b.get(i4);
                                    if (!d0Var5.wasReturnedFromScrap() && d0Var5.getItemId() == itemId) {
                                        d0Var5.addFlags(32);
                                        d0Var = d0Var5;
                                        break;
                                    }
                                }
                            }
                        }
                        if (d0Var != null) {
                            z2 = true;
                            if (d0Var == null) {
                                int size2 = this.f434a.size();
                                for (int i5 = 0; i5 < size2; i5++) {
                                    d0Var4 = this.f434a.get(i5);
                                    if (!d0Var4.wasReturnedFromScrap() && d0Var4.getLayoutPosition() == i && !d0Var4.isInvalid() && (RecyclerView.this.mState.f396g || !d0Var4.isRemoved())) {
                                        d0Var4.addFlags(32);
                                        break;
                                    }
                                }
                                if (!z) {
                                    b.w.b.b bVar = RecyclerView.this.mChildHelper;
                                    int size3 = bVar.f2714c.size();
                                    int i6 = 0;
                                    while (true) {
                                        if (i6 >= size3) {
                                            view = null;
                                            break;
                                        }
                                        view = bVar.f2714c.get(i6);
                                        Objects.requireNonNull((e) bVar.f2712a);
                                        d0 childViewHolderInt = RecyclerView.getChildViewHolderInt(view);
                                        if (childViewHolderInt.getLayoutPosition() == i && !childViewHolderInt.isInvalid() && !childViewHolderInt.isRemoved()) {
                                            break;
                                        }
                                        i6++;
                                    }
                                    if (view != null) {
                                        d0Var = RecyclerView.getChildViewHolderInt(view);
                                        b.w.b.b bVar2 = RecyclerView.this.mChildHelper;
                                        int indexOfChild = RecyclerView.this.indexOfChild(view);
                                        if (indexOfChild >= 0) {
                                            if (bVar2.f2713b.d(indexOfChild)) {
                                                bVar2.f2713b.a(indexOfChild);
                                                bVar2.m(view);
                                                int j2 = RecyclerView.this.mChildHelper.j(view);
                                                if (j2 != -1) {
                                                    RecyclerView.this.mChildHelper.c(j2);
                                                    j(view);
                                                    d0Var.addFlags(8224);
                                                    if (d0Var != null) {
                                                        if (d0Var.isRemoved()) {
                                                            z6 = RecyclerView.this.mState.f396g;
                                                        } else {
                                                            int i7 = d0Var.mPosition;
                                                            if (i7 >= 0 && i7 < RecyclerView.this.mAdapter.getItemCount()) {
                                                                RecyclerView recyclerView = RecyclerView.this;
                                                                z6 = (recyclerView.mState.f396g || recyclerView.mAdapter.getItemViewType(d0Var.mPosition) == d0Var.getItemViewType()) && (!RecyclerView.this.mAdapter.hasStableIds() || d0Var.getItemId() == RecyclerView.this.mAdapter.getItemId(d0Var.mPosition));
                                                            } else {
                                                                StringBuilder sb = new StringBuilder();
                                                                sb.append("Inconsistency detected. Invalid view holder adapter position");
                                                                sb.append(d0Var);
                                                                throw new IndexOutOfBoundsException(c.b.a.a.a.i(RecyclerView.this, sb));
                                                            }
                                                        }
                                                        if (z6) {
                                                            z2 = true;
                                                        } else {
                                                            if (!z) {
                                                                d0Var.addFlags(4);
                                                                if (d0Var.isScrap()) {
                                                                    RecyclerView.this.removeDetachedView(d0Var.itemView, false);
                                                                    d0Var.unScrap();
                                                                } else if (d0Var.wasReturnedFromScrap()) {
                                                                    d0Var.clearReturnedFromScrapFlag();
                                                                }
                                                                i(d0Var);
                                                            }
                                                            d0Var = null;
                                                        }
                                                    }
                                                } else {
                                                    StringBuilder sb2 = new StringBuilder();
                                                    sb2.append("layout index should not be -1 after unhiding a view:");
                                                    sb2.append(d0Var);
                                                    throw new IllegalStateException(c.b.a.a.a.i(RecyclerView.this, sb2));
                                                }
                                            } else {
                                                throw new RuntimeException("trying to unhide a view that was not hidden" + view);
                                            }
                                        } else {
                                            throw new IllegalArgumentException("view is not a child, cannot hide " + view);
                                        }
                                    }
                                }
                                int size4 = this.f436c.size();
                                for (int i8 = 0; i8 < size4; i8++) {
                                    d0Var4 = this.f436c.get(i8);
                                    if (!d0Var4.isInvalid() && d0Var4.getLayoutPosition() == i && !d0Var4.isAttachedToTransitionOverlay()) {
                                        if (!z) {
                                            this.f436c.remove(i8);
                                        }
                                        d0Var = d0Var4;
                                        if (d0Var != null) {
                                        }
                                    }
                                }
                                d0Var = null;
                                if (d0Var != null) {
                                }
                            }
                            if (d0Var == null) {
                                int f3 = RecyclerView.this.mAdapterHelper.f(i, 0);
                                if (f3 >= 0 && f3 < RecyclerView.this.mAdapter.getItemCount()) {
                                    int itemViewType = RecyclerView.this.mAdapter.getItemViewType(f3);
                                    if (RecyclerView.this.mAdapter.hasStableIds()) {
                                        long itemId2 = RecyclerView.this.mAdapter.getItemId(f3);
                                        int size5 = this.f434a.size() - 1;
                                        while (true) {
                                            if (size5 >= 0) {
                                                d0Var3 = this.f434a.get(size5);
                                                if (d0Var3.getItemId() == itemId2 && !d0Var3.wasReturnedFromScrap()) {
                                                    if (itemViewType == d0Var3.getItemViewType()) {
                                                        d0Var3.addFlags(i2);
                                                        if (d0Var3.isRemoved() && !RecyclerView.this.mState.f396g) {
                                                            d0Var3.setFlags(2, 14);
                                                        }
                                                    } else if (!z) {
                                                        this.f434a.remove(size5);
                                                        RecyclerView.this.removeDetachedView(d0Var3.itemView, false);
                                                        d0 childViewHolderInt2 = RecyclerView.getChildViewHolderInt(d0Var3.itemView);
                                                        childViewHolderInt2.mScrapContainer = null;
                                                        childViewHolderInt2.mInChangeScrap = false;
                                                        childViewHolderInt2.clearReturnedFromScrapFlag();
                                                        i(childViewHolderInt2);
                                                    }
                                                }
                                                size5--;
                                                i2 = 32;
                                            } else {
                                                int size6 = this.f436c.size() - 1;
                                                while (true) {
                                                    if (size6 < 0) {
                                                        break;
                                                    }
                                                    d0Var3 = this.f436c.get(size6);
                                                    if (d0Var3.getItemId() == itemId2 && !d0Var3.isAttachedToTransitionOverlay()) {
                                                        if (itemViewType == d0Var3.getItemViewType()) {
                                                            if (!z) {
                                                                this.f436c.remove(size6);
                                                            }
                                                        } else if (!z) {
                                                            g(size6);
                                                            break;
                                                        }
                                                    }
                                                    size6--;
                                                }
                                            }
                                        }
                                        if (d0Var != null) {
                                            d0Var.mPosition = f3;
                                            z2 = true;
                                        }
                                    }
                                    if (d0Var == null) {
                                        u.a aVar = d().f428a.get(itemViewType);
                                        if (aVar != null && !aVar.f430a.isEmpty()) {
                                            ArrayList<d0> arrayList2 = aVar.f430a;
                                            for (int size7 = arrayList2.size() - 1; size7 >= 0; size7--) {
                                                if (!arrayList2.get(size7).isAttachedToTransitionOverlay()) {
                                                    d0Var2 = arrayList2.remove(size7);
                                                    break;
                                                }
                                            }
                                        }
                                        d0Var2 = null;
                                        if (d0Var2 != null) {
                                            d0Var2.resetInternal();
                                            if (RecyclerView.FORCE_INVALIDATE_DISPLAY_LIST) {
                                                View view2 = d0Var2.itemView;
                                                if (view2 instanceof ViewGroup) {
                                                    e((ViewGroup) view2, false);
                                                }
                                            }
                                        }
                                        d0Var = d0Var2;
                                    }
                                    if (d0Var == null) {
                                        long nanoTime = RecyclerView.this.getNanoTime();
                                        if (j != RecyclerView.FOREVER_NS) {
                                            long j3 = this.f440g.a(itemViewType).f432c;
                                            if (!(j3 == 0 || j3 + nanoTime < j)) {
                                                return null;
                                            }
                                        }
                                        RecyclerView recyclerView2 = RecyclerView.this;
                                        d0 createViewHolder = recyclerView2.mAdapter.createViewHolder(recyclerView2, itemViewType);
                                        if (RecyclerView.ALLOW_THREAD_GAP_WORK && (findNestedRecyclerView = RecyclerView.findNestedRecyclerView(createViewHolder.itemView)) != null) {
                                            createViewHolder.mNestedRecyclerView = new WeakReference<>(findNestedRecyclerView);
                                        }
                                        long nanoTime2 = RecyclerView.this.getNanoTime();
                                        u uVar = this.f440g;
                                        long j4 = nanoTime2 - nanoTime;
                                        u.a a2 = uVar.a(itemViewType);
                                        a2.f432c = uVar.b(a2.f432c, j4);
                                        d0Var = createViewHolder;
                                    }
                                } else {
                                    StringBuilder z8 = c.b.a.a.a.z("Inconsistency detected. Invalid item position ", i, "(offset:", f3, ").state:");
                                    z8.append(RecyclerView.this.mState.b());
                                    throw new IndexOutOfBoundsException(c.b.a.a.a.i(RecyclerView.this, z8));
                                }
                            }
                            if (z2 && !RecyclerView.this.mState.f396g && d0Var.hasAnyOfTheFlags(8192)) {
                                d0Var.setFlags(0, 8192);
                                if (RecyclerView.this.mState.j) {
                                    l.b(d0Var);
                                    RecyclerView recyclerView3 = RecyclerView.this;
                                    l lVar = recyclerView3.mItemAnimator;
                                    a0 a0Var = recyclerView3.mState;
                                    d0Var.getUnmodifiedPayloads();
                                    RecyclerView.this.recordAnimationInfoIfBouncedHiddenView(d0Var, lVar.h(d0Var));
                                }
                            }
                            if (!RecyclerView.this.mState.f396g && d0Var.isBound()) {
                                d0Var.mPreLayoutPosition = i;
                            } else if (d0Var.isBound() || d0Var.needsUpdate() || d0Var.isInvalid()) {
                                int f4 = RecyclerView.this.mAdapterHelper.f(i, 0);
                                d0Var.mOwnerRecyclerView = RecyclerView.this;
                                int itemViewType2 = d0Var.getItemViewType();
                                long nanoTime3 = RecyclerView.this.getNanoTime();
                                if (j != RecyclerView.FOREVER_NS) {
                                    long j5 = this.f440g.a(itemViewType2).f433d;
                                }
                                RecyclerView.this.mAdapter.bindViewHolder(d0Var, f4);
                                long nanoTime4 = RecyclerView.this.getNanoTime();
                                u uVar2 = this.f440g;
                                u.a a3 = uVar2.a(d0Var.getItemViewType());
                                a3.f433d = uVar2.b(a3.f433d, nanoTime4 - nanoTime3);
                                if (RecyclerView.this.isAccessibilityEnabled()) {
                                    z3 = true;
                                } else {
                                    View view3 = d0Var.itemView;
                                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                                    if (view3.getImportantForAccessibility() == 0) {
                                        z5 = true;
                                        view3.setImportantForAccessibility(1);
                                    } else {
                                        z5 = true;
                                    }
                                    b.w.b.v vVar = RecyclerView.this.mAccessibilityDelegate;
                                    if (vVar != null) {
                                        b.j.j.a itemDelegate = vVar.getItemDelegate();
                                        if (itemDelegate instanceof v.a) {
                                            v.a aVar2 = (v.a) itemDelegate;
                                            Objects.requireNonNull(aVar2);
                                            b.j.j.a e2 = b.j.j.q.e(view3);
                                            if (e2 != null && e2 != aVar2) {
                                                aVar2.f2801b.put(view3, e2);
                                            }
                                        }
                                        b.j.j.q.n(view3, itemDelegate);
                                    }
                                    z3 = z5;
                                }
                                if (RecyclerView.this.mState.f396g) {
                                    d0Var.mPreLayoutPosition = i;
                                }
                                z4 = z3;
                                layoutParams = d0Var.itemView.getLayoutParams();
                                if (layoutParams == null) {
                                    pVar = (p) RecyclerView.this.generateDefaultLayoutParams();
                                    d0Var.itemView.setLayoutParams(pVar);
                                } else if (!RecyclerView.this.checkLayoutParams(layoutParams)) {
                                    pVar = (p) RecyclerView.this.generateLayoutParams(layoutParams);
                                    d0Var.itemView.setLayoutParams(pVar);
                                } else {
                                    pVar = (p) layoutParams;
                                }
                                pVar.f424a = d0Var;
                                if (z2 && z3) {
                                    z7 = z4;
                                }
                                pVar.f427d = z7;
                                return d0Var;
                            }
                            z4 = true;
                            z3 = false;
                            layoutParams = d0Var.itemView.getLayoutParams();
                            if (layoutParams == null) {
                            }
                            pVar.f424a = d0Var;
                            if (z2) {
                                z7 = z4;
                            }
                            pVar.f427d = z7;
                            return d0Var;
                        }
                    }
                    d0Var = null;
                    if (d0Var != null) {
                    }
                } else {
                    d0Var = null;
                }
                z2 = false;
                if (d0Var == null) {
                }
                if (d0Var == null) {
                }
                if (z2) {
                    d0Var.setFlags(0, 8192);
                    if (RecyclerView.this.mState.j) {
                    }
                }
                if (!RecyclerView.this.mState.f396g) {
                }
                if (d0Var.isBound()) {
                }
                int f42 = RecyclerView.this.mAdapterHelper.f(i, 0);
                d0Var.mOwnerRecyclerView = RecyclerView.this;
                int itemViewType22 = d0Var.getItemViewType();
                long nanoTime32 = RecyclerView.this.getNanoTime();
                if (j != RecyclerView.FOREVER_NS) {
                }
                RecyclerView.this.mAdapter.bindViewHolder(d0Var, f42);
                long nanoTime42 = RecyclerView.this.getNanoTime();
                u uVar22 = this.f440g;
                u.a a32 = uVar22.a(d0Var.getItemViewType());
                a32.f433d = uVar22.b(a32.f433d, nanoTime42 - nanoTime32);
                if (RecyclerView.this.isAccessibilityEnabled()) {
                }
                if (RecyclerView.this.mState.f396g) {
                }
                z4 = z3;
                layoutParams = d0Var.itemView.getLayoutParams();
                if (layoutParams == null) {
                }
                pVar.f424a = d0Var;
                if (z2) {
                }
                pVar.f427d = z7;
                return d0Var;
            }
            StringBuilder z9 = c.b.a.a.a.z("Invalid item position ", i, "(", i, "). Item count:");
            z9.append(RecyclerView.this.mState.b());
            throw new IndexOutOfBoundsException(c.b.a.a.a.i(RecyclerView.this, z9));
        }

        public void l(d0 d0Var) {
            if (d0Var.mInChangeScrap) {
                this.f435b.remove(d0Var);
            } else {
                this.f434a.remove(d0Var);
            }
            d0Var.mScrapContainer = null;
            d0Var.mInChangeScrap = false;
            d0Var.clearReturnedFromScrapFlag();
        }

        public void m() {
            o oVar = RecyclerView.this.mLayout;
            this.f439f = this.f438e + (oVar != null ? oVar.mPrefetchMaxCountObserved : 0);
            for (int size = this.f436c.size() - 1; size >= 0 && this.f436c.size() > this.f439f; size--) {
                g(size);
            }
        }
    }

    /* loaded from: classes.dex */
    public interface w {
        void a(d0 d0Var);
    }

    /* loaded from: classes.dex */
    public class x extends i {
        public x() {
        }

        public void a() {
            if (RecyclerView.POST_UPDATES_ON_ANIMATION) {
                RecyclerView recyclerView = RecyclerView.this;
                if (recyclerView.mHasFixedSize && recyclerView.mIsAttached) {
                    Runnable runnable = recyclerView.mUpdateChildViewsRunnable;
                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                    recyclerView.postOnAnimation(runnable);
                    return;
                }
            }
            RecyclerView recyclerView2 = RecyclerView.this;
            recyclerView2.mAdapterUpdateDuringMeasure = true;
            recyclerView2.requestLayout();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.i
        public void onChanged() {
            RecyclerView.this.assertNotInLayoutOrScroll(null);
            RecyclerView recyclerView = RecyclerView.this;
            recyclerView.mState.f395f = true;
            recyclerView.processDataSetCompletelyChanged(true);
            if (RecyclerView.this.mAdapterHelper.g()) {
                return;
            }
            RecyclerView.this.requestLayout();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.i
        public void onItemRangeChanged(int i, int i2, Object obj) {
            RecyclerView.this.assertNotInLayoutOrScroll(null);
            b.w.b.a aVar = RecyclerView.this.mAdapterHelper;
            Objects.requireNonNull(aVar);
            boolean z = false;
            if (i2 >= 1) {
                aVar.f2703b.add(aVar.h(4, i, i2, obj));
                aVar.f2707f |= 4;
                if (aVar.f2703b.size() == 1) {
                    z = true;
                }
            }
            if (z) {
                a();
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.i
        public void onItemRangeInserted(int i, int i2) {
            RecyclerView.this.assertNotInLayoutOrScroll(null);
            b.w.b.a aVar = RecyclerView.this.mAdapterHelper;
            Objects.requireNonNull(aVar);
            boolean z = false;
            if (i2 >= 1) {
                aVar.f2703b.add(aVar.h(1, i, i2, null));
                aVar.f2707f |= 1;
                if (aVar.f2703b.size() == 1) {
                    z = true;
                }
            }
            if (z) {
                a();
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.i
        public void onItemRangeMoved(int i, int i2, int i3) {
            RecyclerView.this.assertNotInLayoutOrScroll(null);
            b.w.b.a aVar = RecyclerView.this.mAdapterHelper;
            Objects.requireNonNull(aVar);
            boolean z = false;
            if (i != i2) {
                if (i3 == 1) {
                    aVar.f2703b.add(aVar.h(8, i, i2, null));
                    aVar.f2707f |= 8;
                    if (aVar.f2703b.size() == 1) {
                        z = true;
                    }
                } else {
                    throw new IllegalArgumentException("Moving more than 1 item is not supported yet");
                }
            }
            if (z) {
                a();
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.i
        public void onItemRangeRemoved(int i, int i2) {
            RecyclerView.this.assertNotInLayoutOrScroll(null);
            b.w.b.a aVar = RecyclerView.this.mAdapterHelper;
            Objects.requireNonNull(aVar);
            boolean z = false;
            if (i2 >= 1) {
                aVar.f2703b.add(aVar.h(2, i, i2, null));
                aVar.f2707f |= 2;
                if (aVar.f2703b.size() == 1) {
                    z = true;
                }
            }
            if (z) {
                a();
            }
        }
    }

    /* loaded from: classes.dex */
    public static abstract class z {
        private o mLayoutManager;
        private boolean mPendingInitialRun;
        private RecyclerView mRecyclerView;
        private boolean mRunning;
        private boolean mStarted;
        private View mTargetView;
        private int mTargetPosition = -1;
        private final a mRecyclingAction = new a(0, 0);

        /* loaded from: classes.dex */
        public static class a {

            /* renamed from: a  reason: collision with root package name */
            public int f444a;

            /* renamed from: b  reason: collision with root package name */
            public int f445b;

            /* renamed from: d  reason: collision with root package name */
            public int f447d = -1;

            /* renamed from: f  reason: collision with root package name */
            public boolean f449f = false;

            /* renamed from: g  reason: collision with root package name */
            public int f450g = 0;

            /* renamed from: c  reason: collision with root package name */
            public int f446c = Integer.MIN_VALUE;

            /* renamed from: e  reason: collision with root package name */
            public Interpolator f448e = null;

            public a(int i, int i2) {
                this.f444a = i;
                this.f445b = i2;
            }

            public void a(RecyclerView recyclerView) {
                int i = this.f447d;
                if (i >= 0) {
                    this.f447d = -1;
                    recyclerView.jumpToPositionForSmoothScroller(i);
                    this.f449f = false;
                } else if (this.f449f) {
                    Interpolator interpolator = this.f448e;
                    if (interpolator != null && this.f446c < 1) {
                        throw new IllegalStateException("If you provide an interpolator, you must set a positive duration");
                    }
                    int i2 = this.f446c;
                    if (i2 >= 1) {
                        recyclerView.mViewFlinger.b(this.f444a, this.f445b, i2, interpolator);
                        int i3 = this.f450g + 1;
                        this.f450g = i3;
                        if (i3 > 10) {
                            Log.e(RecyclerView.TAG, "Smooth Scroll action is being updated too frequently. Make sure you are not changing it unless necessary");
                        }
                        this.f449f = false;
                        return;
                    }
                    throw new IllegalStateException("Scroll duration must be a positive number");
                } else {
                    this.f450g = 0;
                }
            }

            public void b(int i, int i2, int i3, Interpolator interpolator) {
                this.f444a = i;
                this.f445b = i2;
                this.f446c = i3;
                this.f448e = interpolator;
                this.f449f = true;
            }
        }

        /* loaded from: classes.dex */
        public interface b {
            PointF computeScrollVectorForPosition(int i);
        }

        public PointF computeScrollVectorForPosition(int i) {
            o layoutManager = getLayoutManager();
            if (layoutManager instanceof b) {
                return ((b) layoutManager).computeScrollVectorForPosition(i);
            }
            StringBuilder x = c.b.a.a.a.x("You should override computeScrollVectorForPosition when the LayoutManager does not implement ");
            x.append(b.class.getCanonicalName());
            Log.w(RecyclerView.TAG, x.toString());
            return null;
        }

        public View findViewByPosition(int i) {
            return this.mRecyclerView.mLayout.findViewByPosition(i);
        }

        public int getChildCount() {
            return this.mRecyclerView.mLayout.getChildCount();
        }

        public int getChildPosition(View view) {
            return this.mRecyclerView.getChildLayoutPosition(view);
        }

        public o getLayoutManager() {
            return this.mLayoutManager;
        }

        public int getTargetPosition() {
            return this.mTargetPosition;
        }

        @Deprecated
        public void instantScrollToPosition(int i) {
            this.mRecyclerView.scrollToPosition(i);
        }

        public boolean isPendingInitialRun() {
            return this.mPendingInitialRun;
        }

        public boolean isRunning() {
            return this.mRunning;
        }

        public void normalize(PointF pointF) {
            float f2 = pointF.x;
            float f3 = pointF.y;
            float sqrt = (float) Math.sqrt((f3 * f3) + (f2 * f2));
            pointF.x /= sqrt;
            pointF.y /= sqrt;
        }

        public void onAnimation(int i, int i2) {
            PointF computeScrollVectorForPosition;
            RecyclerView recyclerView = this.mRecyclerView;
            if (this.mTargetPosition == -1 || recyclerView == null) {
                stop();
            }
            if (this.mPendingInitialRun && this.mTargetView == null && this.mLayoutManager != null && (computeScrollVectorForPosition = computeScrollVectorForPosition(this.mTargetPosition)) != null) {
                float f2 = computeScrollVectorForPosition.x;
                if (f2 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || computeScrollVectorForPosition.y != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    recyclerView.scrollStep((int) Math.signum(f2), (int) Math.signum(computeScrollVectorForPosition.y), null);
                }
            }
            this.mPendingInitialRun = false;
            View view = this.mTargetView;
            if (view != null) {
                if (getChildPosition(view) == this.mTargetPosition) {
                    onTargetFound(this.mTargetView, recyclerView.mState, this.mRecyclingAction);
                    this.mRecyclingAction.a(recyclerView);
                    stop();
                } else {
                    Log.e(RecyclerView.TAG, "Passed over target position while smooth scrolling.");
                    this.mTargetView = null;
                }
            }
            if (this.mRunning) {
                onSeekTargetStep(i, i2, recyclerView.mState, this.mRecyclingAction);
                a aVar = this.mRecyclingAction;
                boolean z = aVar.f447d >= 0;
                aVar.a(recyclerView);
                if (z && this.mRunning) {
                    this.mPendingInitialRun = true;
                    recyclerView.mViewFlinger.a();
                }
            }
        }

        public void onChildAttachedToWindow(View view) {
            if (getChildPosition(view) == getTargetPosition()) {
                this.mTargetView = view;
            }
        }

        public abstract void onSeekTargetStep(int i, int i2, a0 a0Var, a aVar);

        public abstract void onStart();

        public abstract void onStop();

        public abstract void onTargetFound(View view, a0 a0Var, a aVar);

        public void setTargetPosition(int i) {
            this.mTargetPosition = i;
        }

        public void start(RecyclerView recyclerView, o oVar) {
            recyclerView.mViewFlinger.c();
            if (this.mStarted) {
                StringBuilder x = c.b.a.a.a.x("An instance of ");
                x.append(getClass().getSimpleName());
                x.append(" was started more than once. Each instance of");
                x.append(getClass().getSimpleName());
                x.append(" is intended to only be used once. You should create a new instance for each use.");
                Log.w(RecyclerView.TAG, x.toString());
            }
            this.mRecyclerView = recyclerView;
            this.mLayoutManager = oVar;
            int i = this.mTargetPosition;
            if (i != -1) {
                recyclerView.mState.f390a = i;
                this.mRunning = true;
                this.mPendingInitialRun = true;
                this.mTargetView = findViewByPosition(getTargetPosition());
                onStart();
                this.mRecyclerView.mViewFlinger.a();
                this.mStarted = true;
                return;
            }
            throw new IllegalArgumentException("Invalid target position");
        }

        public final void stop() {
            if (this.mRunning) {
                this.mRunning = false;
                onStop();
                this.mRecyclerView.mState.f390a = -1;
                this.mTargetView = null;
                this.mTargetPosition = -1;
                this.mPendingInitialRun = false;
                this.mLayoutManager.onSmoothScrollerStopped(this);
                this.mLayoutManager = null;
                this.mRecyclerView = null;
            }
        }
    }

    static {
        Class<?> cls = Integer.TYPE;
        LAYOUT_MANAGER_CONSTRUCTOR_SIGNATURE = new Class[]{Context.class, AttributeSet.class, cls, cls};
        sQuinticInterpolator = new c();
    }

    public RecyclerView(Context context) {
        this(context, null);
    }

    private void addAnimatingView(d0 d0Var) {
        View view = d0Var.itemView;
        boolean z2 = view.getParent() == this;
        this.mRecycler.l(getChildViewHolder(view));
        if (d0Var.isTmpDetached()) {
            this.mChildHelper.b(view, -1, view.getLayoutParams(), true);
        } else if (!z2) {
            this.mChildHelper.a(view, -1, true);
        } else {
            b.w.b.b bVar = this.mChildHelper;
            int indexOfChild = RecyclerView.this.indexOfChild(view);
            if (indexOfChild >= 0) {
                bVar.f2713b.h(indexOfChild);
                bVar.i(view);
                return;
            }
            throw new IllegalArgumentException("view is not a child, cannot hide " + view);
        }
    }

    private void animateChange(d0 d0Var, d0 d0Var2, l.c cVar, l.c cVar2, boolean z2, boolean z3) {
        d0Var.setIsRecyclable(false);
        if (z2) {
            addAnimatingView(d0Var);
        }
        if (d0Var != d0Var2) {
            if (z3) {
                addAnimatingView(d0Var2);
            }
            d0Var.mShadowedHolder = d0Var2;
            addAnimatingView(d0Var);
            this.mRecycler.l(d0Var);
            d0Var2.setIsRecyclable(false);
            d0Var2.mShadowingHolder = d0Var;
        }
        if (this.mItemAnimator.a(d0Var, d0Var2, cVar, cVar2)) {
            postAnimationRunner();
        }
    }

    private void cancelScroll() {
        resetScroll();
        setScrollState(0);
    }

    public static void clearNestedRecyclerViewIfNotNested(d0 d0Var) {
        WeakReference<RecyclerView> weakReference = d0Var.mNestedRecyclerView;
        if (weakReference != null) {
            RecyclerView recyclerView = weakReference.get();
            while (recyclerView != null) {
                if (recyclerView == d0Var.itemView) {
                    return;
                }
                ViewParent parent = recyclerView.getParent();
                recyclerView = parent instanceof View ? (View) parent : null;
            }
            d0Var.mNestedRecyclerView = null;
        }
    }

    private void createLayoutManager(Context context, String str, AttributeSet attributeSet, int i2, int i3) {
        ClassLoader classLoader;
        Constructor constructor;
        if (str != null) {
            String trim = str.trim();
            if (trim.isEmpty()) {
                return;
            }
            String fullClassName = getFullClassName(context, trim);
            try {
                if (isInEditMode()) {
                    classLoader = getClass().getClassLoader();
                } else {
                    classLoader = context.getClassLoader();
                }
                Class<? extends U> asSubclass = Class.forName(fullClassName, false, classLoader).asSubclass(o.class);
                Object[] objArr = null;
                try {
                    constructor = asSubclass.getConstructor(LAYOUT_MANAGER_CONSTRUCTOR_SIGNATURE);
                    objArr = new Object[]{context, attributeSet, Integer.valueOf(i2), Integer.valueOf(i3)};
                } catch (NoSuchMethodException e2) {
                    try {
                        constructor = asSubclass.getConstructor(new Class[0]);
                    } catch (NoSuchMethodException e3) {
                        e3.initCause(e2);
                        throw new IllegalStateException(attributeSet.getPositionDescription() + ": Error creating LayoutManager " + fullClassName, e3);
                    }
                }
                constructor.setAccessible(true);
                setLayoutManager((o) constructor.newInstance(objArr));
            } catch (ClassCastException e4) {
                throw new IllegalStateException(attributeSet.getPositionDescription() + ": Class is not a LayoutManager " + fullClassName, e4);
            } catch (ClassNotFoundException e5) {
                throw new IllegalStateException(attributeSet.getPositionDescription() + ": Unable to find LayoutManager " + fullClassName, e5);
            } catch (IllegalAccessException e6) {
                throw new IllegalStateException(attributeSet.getPositionDescription() + ": Cannot access non-public constructor " + fullClassName, e6);
            } catch (InstantiationException e7) {
                throw new IllegalStateException(attributeSet.getPositionDescription() + ": Could not instantiate the LayoutManager: " + fullClassName, e7);
            } catch (InvocationTargetException e8) {
                throw new IllegalStateException(attributeSet.getPositionDescription() + ": Could not instantiate the LayoutManager: " + fullClassName, e8);
            }
        }
    }

    private boolean didChildRangeChange(int i2, int i3) {
        findMinMaxChildLayoutPositions(this.mMinMaxLayoutPositions);
        int[] iArr = this.mMinMaxLayoutPositions;
        return (iArr[0] == i2 && iArr[1] == i3) ? false : true;
    }

    private void dispatchContentChangedIfNecessary() {
        int i2 = this.mEatenAccessibilityChangeFlags;
        this.mEatenAccessibilityChangeFlags = 0;
        if (i2 == 0 || !isAccessibilityEnabled()) {
            return;
        }
        AccessibilityEvent obtain = AccessibilityEvent.obtain();
        obtain.setEventType(2048);
        obtain.setContentChangeTypes(i2);
        sendAccessibilityEventUnchecked(obtain);
    }

    private void dispatchLayoutStep1() {
        this.mState.a(1);
        fillRemainingScrollValues(this.mState);
        this.mState.i = false;
        startInterceptRequestLayout();
        b.w.b.z zVar = this.mViewInfoStore;
        zVar.f2814a.clear();
        zVar.f2815b.a();
        onEnterLayoutOrScroll();
        processAdapterUpdatesAndSetAnimationFlags();
        saveFocusInfo();
        a0 a0Var = this.mState;
        a0Var.f397h = a0Var.j && this.mItemsChanged;
        this.mItemsChanged = false;
        this.mItemsAddedOrRemoved = false;
        a0Var.f396g = a0Var.k;
        a0Var.f394e = this.mAdapter.getItemCount();
        findMinMaxChildLayoutPositions(this.mMinMaxLayoutPositions);
        if (this.mState.j) {
            int e2 = this.mChildHelper.e();
            for (int i2 = 0; i2 < e2; i2++) {
                d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.d(i2));
                if (!childViewHolderInt.shouldIgnore() && (!childViewHolderInt.isInvalid() || this.mAdapter.hasStableIds())) {
                    l lVar = this.mItemAnimator;
                    l.b(childViewHolderInt);
                    childViewHolderInt.getUnmodifiedPayloads();
                    this.mViewInfoStore.c(childViewHolderInt, lVar.h(childViewHolderInt));
                    if (this.mState.f397h && childViewHolderInt.isUpdated() && !childViewHolderInt.isRemoved() && !childViewHolderInt.shouldIgnore() && !childViewHolderInt.isInvalid()) {
                        this.mViewInfoStore.f2815b.g(getChangedHolderKey(childViewHolderInt), childViewHolderInt);
                    }
                }
            }
        }
        if (this.mState.k) {
            saveOldPositions();
            a0 a0Var2 = this.mState;
            boolean z2 = a0Var2.f395f;
            a0Var2.f395f = false;
            this.mLayout.onLayoutChildren(this.mRecycler, a0Var2);
            this.mState.f395f = z2;
            for (int i3 = 0; i3 < this.mChildHelper.e(); i3++) {
                d0 childViewHolderInt2 = getChildViewHolderInt(this.mChildHelper.d(i3));
                if (!childViewHolderInt2.shouldIgnore()) {
                    z.a orDefault = this.mViewInfoStore.f2814a.getOrDefault(childViewHolderInt2, null);
                    if (!((orDefault == null || (orDefault.f2817b & 4) == 0) ? false : true)) {
                        l.b(childViewHolderInt2);
                        boolean hasAnyOfTheFlags = childViewHolderInt2.hasAnyOfTheFlags(8192);
                        l lVar2 = this.mItemAnimator;
                        childViewHolderInt2.getUnmodifiedPayloads();
                        l.c h2 = lVar2.h(childViewHolderInt2);
                        if (hasAnyOfTheFlags) {
                            recordAnimationInfoIfBouncedHiddenView(childViewHolderInt2, h2);
                        } else {
                            b.w.b.z zVar2 = this.mViewInfoStore;
                            z.a orDefault2 = zVar2.f2814a.getOrDefault(childViewHolderInt2, null);
                            if (orDefault2 == null) {
                                orDefault2 = z.a.a();
                                zVar2.f2814a.put(childViewHolderInt2, orDefault2);
                            }
                            orDefault2.f2817b |= 2;
                            orDefault2.f2818c = h2;
                        }
                    }
                }
            }
            clearOldPositions();
        } else {
            clearOldPositions();
        }
        onExitLayoutOrScroll();
        stopInterceptRequestLayout(false);
        this.mState.f393d = 2;
    }

    private void dispatchLayoutStep2() {
        startInterceptRequestLayout();
        onEnterLayoutOrScroll();
        this.mState.a(6);
        this.mAdapterHelper.c();
        this.mState.f394e = this.mAdapter.getItemCount();
        a0 a0Var = this.mState;
        a0Var.f392c = 0;
        a0Var.f396g = false;
        this.mLayout.onLayoutChildren(this.mRecycler, a0Var);
        a0 a0Var2 = this.mState;
        a0Var2.f395f = false;
        this.mPendingSavedState = null;
        a0Var2.j = a0Var2.j && this.mItemAnimator != null;
        a0Var2.f393d = 4;
        onExitLayoutOrScroll();
        stopInterceptRequestLayout(false);
    }

    private void dispatchLayoutStep3() {
        boolean i2;
        this.mState.a(4);
        startInterceptRequestLayout();
        onEnterLayoutOrScroll();
        a0 a0Var = this.mState;
        a0Var.f393d = 1;
        if (a0Var.j) {
            for (int e2 = this.mChildHelper.e() - 1; e2 >= 0; e2--) {
                d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.d(e2));
                if (!childViewHolderInt.shouldIgnore()) {
                    long changedHolderKey = getChangedHolderKey(childViewHolderInt);
                    Objects.requireNonNull(this.mItemAnimator);
                    l.c cVar = new l.c();
                    View view = childViewHolderInt.itemView;
                    cVar.f415a = view.getLeft();
                    cVar.f416b = view.getTop();
                    view.getRight();
                    view.getBottom();
                    d0 e3 = this.mViewInfoStore.f2815b.e(changedHolderKey, null);
                    if (e3 != null && !e3.shouldIgnore()) {
                        boolean d2 = this.mViewInfoStore.d(e3);
                        boolean d3 = this.mViewInfoStore.d(childViewHolderInt);
                        if (d2 && e3 == childViewHolderInt) {
                            this.mViewInfoStore.b(childViewHolderInt, cVar);
                        } else {
                            l.c e4 = this.mViewInfoStore.e(e3, 4);
                            this.mViewInfoStore.b(childViewHolderInt, cVar);
                            l.c e5 = this.mViewInfoStore.e(childViewHolderInt, 8);
                            if (e4 == null) {
                                handleMissingPreInfoForChangeError(changedHolderKey, childViewHolderInt, e3);
                            } else {
                                animateChange(e3, childViewHolderInt, e4, e5, d2, d3);
                            }
                        }
                    } else {
                        this.mViewInfoStore.b(childViewHolderInt, cVar);
                    }
                }
            }
            b.w.b.z zVar = this.mViewInfoStore;
            z.b bVar = this.mViewInfoProcessCallback;
            for (int i3 = zVar.f2814a.f1775h - 1; i3 >= 0; i3--) {
                d0 h2 = zVar.f2814a.h(i3);
                z.a j2 = zVar.f2814a.j(i3);
                int i4 = j2.f2817b;
                if ((i4 & 3) == 3) {
                    RecyclerView recyclerView = RecyclerView.this;
                    recyclerView.mLayout.removeAndRecycleView(h2.itemView, recyclerView.mRecycler);
                } else if ((i4 & 1) != 0) {
                    l.c cVar2 = j2.f2818c;
                    if (cVar2 == null) {
                        RecyclerView recyclerView2 = RecyclerView.this;
                        recyclerView2.mLayout.removeAndRecycleView(h2.itemView, recyclerView2.mRecycler);
                    } else {
                        l.c cVar3 = j2.f2819d;
                        d dVar = (d) bVar;
                        RecyclerView.this.mRecycler.l(h2);
                        RecyclerView.this.animateDisappearance(h2, cVar2, cVar3);
                    }
                } else if ((i4 & 14) == 14) {
                    RecyclerView.this.animateAppearance(h2, j2.f2818c, j2.f2819d);
                } else if ((i4 & 12) == 12) {
                    l.c cVar4 = j2.f2818c;
                    l.c cVar5 = j2.f2819d;
                    d dVar2 = (d) bVar;
                    Objects.requireNonNull(dVar2);
                    h2.setIsRecyclable(false);
                    RecyclerView recyclerView3 = RecyclerView.this;
                    if (recyclerView3.mDataSetHasChangedAfterLayout) {
                        if (recyclerView3.mItemAnimator.a(h2, h2, cVar4, cVar5)) {
                            RecyclerView.this.postAnimationRunner();
                        }
                    } else {
                        b.w.b.w wVar = (b.w.b.w) recyclerView3.mItemAnimator;
                        Objects.requireNonNull(wVar);
                        int i5 = cVar4.f415a;
                        int i6 = cVar5.f415a;
                        if (i5 == i6 && cVar4.f416b == cVar5.f416b) {
                            wVar.c(h2);
                            i2 = false;
                        } else {
                            i2 = wVar.i(h2, i5, cVar4.f416b, i6, cVar5.f416b);
                        }
                        if (i2) {
                            RecyclerView.this.postAnimationRunner();
                        }
                    }
                } else if ((i4 & 4) != 0) {
                    l.c cVar6 = j2.f2818c;
                    d dVar3 = (d) bVar;
                    RecyclerView.this.mRecycler.l(h2);
                    RecyclerView.this.animateDisappearance(h2, cVar6, null);
                } else if ((i4 & 8) != 0) {
                    RecyclerView.this.animateAppearance(h2, j2.f2818c, j2.f2819d);
                }
                z.a.b(j2);
            }
        }
        this.mLayout.removeAndRecycleScrapInt(this.mRecycler);
        a0 a0Var2 = this.mState;
        a0Var2.f391b = a0Var2.f394e;
        this.mDataSetHasChangedAfterLayout = false;
        this.mDispatchItemsChangedEvent = false;
        a0Var2.j = false;
        a0Var2.k = false;
        this.mLayout.mRequestedSimpleAnimations = false;
        ArrayList<d0> arrayList = this.mRecycler.f435b;
        if (arrayList != null) {
            arrayList.clear();
        }
        o oVar = this.mLayout;
        if (oVar.mPrefetchMaxObservedInInitialPrefetch) {
            oVar.mPrefetchMaxCountObserved = 0;
            oVar.mPrefetchMaxObservedInInitialPrefetch = false;
            this.mRecycler.m();
        }
        this.mLayout.onLayoutCompleted(this.mState);
        onExitLayoutOrScroll();
        stopInterceptRequestLayout(false);
        b.w.b.z zVar2 = this.mViewInfoStore;
        zVar2.f2814a.clear();
        zVar2.f2815b.a();
        int[] iArr = this.mMinMaxLayoutPositions;
        if (didChildRangeChange(iArr[0], iArr[1])) {
            dispatchOnScrolled(0, 0);
        }
        recoverFocusFromState();
        resetFocusInfo();
    }

    private boolean dispatchToOnItemTouchListeners(MotionEvent motionEvent) {
        s sVar = this.mInterceptingOnItemTouchListener;
        if (sVar == null) {
            if (motionEvent.getAction() == 0) {
                return false;
            }
            return findInterceptingOnItemTouchListener(motionEvent);
        }
        sVar.a(this, motionEvent);
        int action = motionEvent.getAction();
        if (action == 3 || action == 1) {
            this.mInterceptingOnItemTouchListener = null;
        }
        return true;
    }

    private boolean findInterceptingOnItemTouchListener(MotionEvent motionEvent) {
        int action = motionEvent.getAction();
        int size = this.mOnItemTouchListeners.size();
        for (int i2 = 0; i2 < size; i2++) {
            s sVar = this.mOnItemTouchListeners.get(i2);
            if (sVar.b(this, motionEvent) && action != 3) {
                this.mInterceptingOnItemTouchListener = sVar;
                return true;
            }
        }
        return false;
    }

    private void findMinMaxChildLayoutPositions(int[] iArr) {
        int e2 = this.mChildHelper.e();
        if (e2 == 0) {
            iArr[0] = -1;
            iArr[1] = -1;
            return;
        }
        int i2 = Integer.MAX_VALUE;
        int i3 = Integer.MIN_VALUE;
        for (int i4 = 0; i4 < e2; i4++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.d(i4));
            if (!childViewHolderInt.shouldIgnore()) {
                int layoutPosition = childViewHolderInt.getLayoutPosition();
                if (layoutPosition < i2) {
                    i2 = layoutPosition;
                }
                if (layoutPosition > i3) {
                    i3 = layoutPosition;
                }
            }
        }
        iArr[0] = i2;
        iArr[1] = i3;
    }

    public static RecyclerView findNestedRecyclerView(View view) {
        if (view instanceof ViewGroup) {
            if (view instanceof RecyclerView) {
                return (RecyclerView) view;
            }
            ViewGroup viewGroup = (ViewGroup) view;
            int childCount = viewGroup.getChildCount();
            for (int i2 = 0; i2 < childCount; i2++) {
                RecyclerView findNestedRecyclerView = findNestedRecyclerView(viewGroup.getChildAt(i2));
                if (findNestedRecyclerView != null) {
                    return findNestedRecyclerView;
                }
            }
            return null;
        }
        return null;
    }

    private View findNextViewToFocus() {
        d0 findViewHolderForAdapterPosition;
        a0 a0Var = this.mState;
        int i2 = a0Var.l;
        if (i2 == -1) {
            i2 = 0;
        }
        int b2 = a0Var.b();
        for (int i3 = i2; i3 < b2; i3++) {
            d0 findViewHolderForAdapterPosition2 = findViewHolderForAdapterPosition(i3);
            if (findViewHolderForAdapterPosition2 == null) {
                break;
            } else if (findViewHolderForAdapterPosition2.itemView.hasFocusable()) {
                return findViewHolderForAdapterPosition2.itemView;
            }
        }
        int min = Math.min(b2, i2);
        while (true) {
            min--;
            if (min < 0 || (findViewHolderForAdapterPosition = findViewHolderForAdapterPosition(min)) == null) {
                return null;
            }
            if (findViewHolderForAdapterPosition.itemView.hasFocusable()) {
                return findViewHolderForAdapterPosition.itemView;
            }
        }
    }

    public static d0 getChildViewHolderInt(View view) {
        if (view == null) {
            return null;
        }
        return ((p) view.getLayoutParams()).f424a;
    }

    public static void getDecoratedBoundsWithMarginsInt(View view, Rect rect) {
        p pVar = (p) view.getLayoutParams();
        Rect rect2 = pVar.f425b;
        rect.set((view.getLeft() - rect2.left) - ((ViewGroup.MarginLayoutParams) pVar).leftMargin, (view.getTop() - rect2.top) - ((ViewGroup.MarginLayoutParams) pVar).topMargin, view.getRight() + rect2.right + ((ViewGroup.MarginLayoutParams) pVar).rightMargin, view.getBottom() + rect2.bottom + ((ViewGroup.MarginLayoutParams) pVar).bottomMargin);
    }

    private int getDeepestFocusedViewWithId(View view) {
        int id = view.getId();
        while (!view.isFocused() && (view instanceof ViewGroup) && view.hasFocus()) {
            view = ((ViewGroup) view).getFocusedChild();
            if (view.getId() != -1) {
                id = view.getId();
            }
        }
        return id;
    }

    private String getFullClassName(Context context, String str) {
        if (str.charAt(0) == '.') {
            return context.getPackageName() + str;
        } else if (str.contains(".")) {
            return str;
        } else {
            return RecyclerView.class.getPackage().getName() + '.' + str;
        }
    }

    private b.j.j.f getScrollingChildHelper() {
        if (this.mScrollingChildHelper == null) {
            this.mScrollingChildHelper = new b.j.j.f(this);
        }
        return this.mScrollingChildHelper;
    }

    private void handleMissingPreInfoForChangeError(long j2, d0 d0Var, d0 d0Var2) {
        int e2 = this.mChildHelper.e();
        for (int i2 = 0; i2 < e2; i2++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.d(i2));
            if (childViewHolderInt != d0Var && getChangedHolderKey(childViewHolderInt) == j2) {
                g gVar = this.mAdapter;
                if (gVar != null && gVar.hasStableIds()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("Two different ViewHolders have the same stable ID. Stable IDs in your adapter MUST BE unique and SHOULD NOT change.\n ViewHolder 1:");
                    sb.append(childViewHolderInt);
                    sb.append(" \n View Holder 2:");
                    sb.append(d0Var);
                    throw new IllegalStateException(c.b.a.a.a.i(this, sb));
                }
                StringBuilder sb2 = new StringBuilder();
                sb2.append("Two different ViewHolders have the same change ID. This might happen due to inconsistent Adapter update events or if the LayoutManager lays out the same View multiple times.\n ViewHolder 1:");
                sb2.append(childViewHolderInt);
                sb2.append(" \n View Holder 2:");
                sb2.append(d0Var);
                throw new IllegalStateException(c.b.a.a.a.i(this, sb2));
            }
        }
        Log.e(TAG, "Problem while matching changed view holders with the newones. The pre-layout information for the change holder " + d0Var2 + " cannot be found but it is necessary for " + d0Var + exceptionLabel());
    }

    private boolean hasUpdatedView() {
        int e2 = this.mChildHelper.e();
        for (int i2 = 0; i2 < e2; i2++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.d(i2));
            if (childViewHolderInt != null && !childViewHolderInt.shouldIgnore() && childViewHolderInt.isUpdated()) {
                return true;
            }
        }
        return false;
    }

    @SuppressLint({"InlinedApi"})
    private void initAutofill() {
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        int i2 = Build.VERSION.SDK_INT;
        if ((i2 >= 26 ? getImportantForAutofill() : 0) != 0 || i2 < 26) {
            return;
        }
        setImportantForAutofill(8);
    }

    private void initChildrenHelper() {
        this.mChildHelper = new b.w.b.b(new e());
    }

    private boolean isPreferredNextFocus(View view, View view2, int i2) {
        int i3;
        if (view2 == null || view2 == this || findContainingItemView(view2) == null) {
            return false;
        }
        if (view == null || findContainingItemView(view) == null) {
            return true;
        }
        this.mTempRect.set(0, 0, view.getWidth(), view.getHeight());
        this.mTempRect2.set(0, 0, view2.getWidth(), view2.getHeight());
        offsetDescendantRectToMyCoords(view, this.mTempRect);
        offsetDescendantRectToMyCoords(view2, this.mTempRect2);
        char c2 = 65535;
        int i4 = this.mLayout.getLayoutDirection() == 1 ? -1 : 1;
        Rect rect = this.mTempRect;
        int i5 = rect.left;
        Rect rect2 = this.mTempRect2;
        int i6 = rect2.left;
        if ((i5 < i6 || rect.right <= i6) && rect.right < rect2.right) {
            i3 = 1;
        } else {
            int i7 = rect.right;
            int i8 = rect2.right;
            i3 = ((i7 > i8 || i5 >= i8) && i5 > i6) ? -1 : 0;
        }
        int i9 = rect.top;
        int i10 = rect2.top;
        if ((i9 < i10 || rect.bottom <= i10) && rect.bottom < rect2.bottom) {
            c2 = 1;
        } else {
            int i11 = rect.bottom;
            int i12 = rect2.bottom;
            if ((i11 <= i12 && i9 < i12) || i9 <= i10) {
                c2 = 0;
            }
        }
        if (i2 == 1) {
            return c2 < 0 || (c2 == 0 && i3 * i4 <= 0);
        } else if (i2 == 2) {
            return c2 > 0 || (c2 == 0 && i3 * i4 >= 0);
        } else if (i2 == 17) {
            return i3 < 0;
        } else if (i2 == 33) {
            return c2 < 0;
        } else if (i2 == 66) {
            return i3 > 0;
        } else if (i2 == 130) {
            return c2 > 0;
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append("Invalid direction: ");
            sb.append(i2);
            throw new IllegalArgumentException(c.b.a.a.a.i(this, sb));
        }
    }

    private void onPointerUp(MotionEvent motionEvent) {
        int actionIndex = motionEvent.getActionIndex();
        if (motionEvent.getPointerId(actionIndex) == this.mScrollPointerId) {
            int i2 = actionIndex == 0 ? 1 : 0;
            this.mScrollPointerId = motionEvent.getPointerId(i2);
            int x2 = (int) (motionEvent.getX(i2) + 0.5f);
            this.mLastTouchX = x2;
            this.mInitialTouchX = x2;
            int y2 = (int) (motionEvent.getY(i2) + 0.5f);
            this.mLastTouchY = y2;
            this.mInitialTouchY = y2;
        }
    }

    private boolean predictiveItemAnimationsEnabled() {
        return this.mItemAnimator != null && this.mLayout.supportsPredictiveItemAnimations();
    }

    private void processAdapterUpdatesAndSetAnimationFlags() {
        boolean z2;
        boolean z3 = false;
        if (this.mDataSetHasChangedAfterLayout) {
            b.w.b.a aVar = this.mAdapterHelper;
            aVar.l(aVar.f2703b);
            aVar.l(aVar.f2704c);
            aVar.f2707f = 0;
            if (this.mDispatchItemsChangedEvent) {
                this.mLayout.onItemsChanged(this);
            }
        }
        if (predictiveItemAnimationsEnabled()) {
            this.mAdapterHelper.j();
        } else {
            this.mAdapterHelper.c();
        }
        boolean z4 = this.mItemsAddedOrRemoved || this.mItemsChanged;
        this.mState.j = this.mFirstLayoutComplete && this.mItemAnimator != null && ((z2 = this.mDataSetHasChangedAfterLayout) || z4 || this.mLayout.mRequestedSimpleAnimations) && (!z2 || this.mAdapter.hasStableIds());
        a0 a0Var = this.mState;
        if (a0Var.j && z4 && !this.mDataSetHasChangedAfterLayout && predictiveItemAnimationsEnabled()) {
            z3 = true;
        }
        a0Var.k = z3;
    }

    /* JADX WARN: Removed duplicated region for block: B:12:0x0040  */
    /* JADX WARN: Removed duplicated region for block: B:13:0x0056  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void pullGlows(float f2, float f3, float f4, float f5) {
        boolean z2;
        boolean z3 = true;
        if (f3 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            ensureLeftGlow();
            this.mLeftGlow.onPull((-f3) / getWidth(), 1.0f - (f4 / getHeight()));
        } else if (f3 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            z2 = false;
            if (f5 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                ensureTopGlow();
                this.mTopGlow.onPull((-f5) / getHeight(), f2 / getWidth());
            } else if (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                ensureBottomGlow();
                this.mBottomGlow.onPull(f5 / getHeight(), 1.0f - (f2 / getWidth()));
            } else {
                z3 = z2;
            }
            if (z3 && f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                return;
            }
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            postInvalidateOnAnimation();
        } else {
            ensureRightGlow();
            this.mRightGlow.onPull(f3 / getWidth(), f4 / getHeight());
        }
        z2 = true;
        if (f5 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
        }
        if (z3) {
        }
        AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
        postInvalidateOnAnimation();
    }

    private void recoverFocusFromState() {
        View findViewById;
        if (!this.mPreserveFocusAfterLayout || this.mAdapter == null || !hasFocus() || getDescendantFocusability() == 393216) {
            return;
        }
        if (getDescendantFocusability() == 131072 && isFocused()) {
            return;
        }
        if (!isFocused()) {
            View focusedChild = getFocusedChild();
            if (IGNORE_DETACHED_FOCUSED_CHILD && (focusedChild.getParent() == null || !focusedChild.hasFocus())) {
                if (this.mChildHelper.e() == 0) {
                    requestFocus();
                    return;
                }
            } else if (!this.mChildHelper.k(focusedChild)) {
                return;
            }
        }
        View view = null;
        d0 findViewHolderForItemId = (this.mState.m == -1 || !this.mAdapter.hasStableIds()) ? null : findViewHolderForItemId(this.mState.m);
        if (findViewHolderForItemId != null && !this.mChildHelper.k(findViewHolderForItemId.itemView) && findViewHolderForItemId.itemView.hasFocusable()) {
            view = findViewHolderForItemId.itemView;
        } else if (this.mChildHelper.e() > 0) {
            view = findNextViewToFocus();
        }
        if (view != null) {
            int i2 = this.mState.n;
            if (i2 != -1 && (findViewById = view.findViewById(i2)) != null && findViewById.isFocusable()) {
                view = findViewById;
            }
            view.requestFocus();
        }
    }

    private void releaseGlows() {
        boolean z2;
        EdgeEffect edgeEffect = this.mLeftGlow;
        if (edgeEffect != null) {
            edgeEffect.onRelease();
            z2 = this.mLeftGlow.isFinished();
        } else {
            z2 = false;
        }
        EdgeEffect edgeEffect2 = this.mTopGlow;
        if (edgeEffect2 != null) {
            edgeEffect2.onRelease();
            z2 |= this.mTopGlow.isFinished();
        }
        EdgeEffect edgeEffect3 = this.mRightGlow;
        if (edgeEffect3 != null) {
            edgeEffect3.onRelease();
            z2 |= this.mRightGlow.isFinished();
        }
        EdgeEffect edgeEffect4 = this.mBottomGlow;
        if (edgeEffect4 != null) {
            edgeEffect4.onRelease();
            z2 |= this.mBottomGlow.isFinished();
        }
        if (z2) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            postInvalidateOnAnimation();
        }
    }

    private void requestChildOnScreen(View view, View view2) {
        View view3 = view2 != null ? view2 : view;
        this.mTempRect.set(0, 0, view3.getWidth(), view3.getHeight());
        ViewGroup.LayoutParams layoutParams = view3.getLayoutParams();
        if (layoutParams instanceof p) {
            p pVar = (p) layoutParams;
            if (!pVar.f426c) {
                Rect rect = pVar.f425b;
                Rect rect2 = this.mTempRect;
                rect2.left -= rect.left;
                rect2.right += rect.right;
                rect2.top -= rect.top;
                rect2.bottom += rect.bottom;
            }
        }
        if (view2 != null) {
            offsetDescendantRectToMyCoords(view2, this.mTempRect);
            offsetRectIntoDescendantCoords(view, this.mTempRect);
        }
        this.mLayout.requestChildRectangleOnScreen(this, view, this.mTempRect, !this.mFirstLayoutComplete, view2 == null);
    }

    private void resetFocusInfo() {
        a0 a0Var = this.mState;
        a0Var.m = -1L;
        a0Var.l = -1;
        a0Var.n = -1;
    }

    private void resetScroll() {
        VelocityTracker velocityTracker = this.mVelocityTracker;
        if (velocityTracker != null) {
            velocityTracker.clear();
        }
        stopNestedScroll(0);
        releaseGlows();
    }

    private void saveFocusInfo() {
        int adapterPosition;
        View focusedChild = (this.mPreserveFocusAfterLayout && hasFocus() && this.mAdapter != null) ? getFocusedChild() : null;
        d0 findContainingViewHolder = focusedChild != null ? findContainingViewHolder(focusedChild) : null;
        if (findContainingViewHolder == null) {
            resetFocusInfo();
            return;
        }
        this.mState.m = this.mAdapter.hasStableIds() ? findContainingViewHolder.getItemId() : -1L;
        a0 a0Var = this.mState;
        if (this.mDataSetHasChangedAfterLayout) {
            adapterPosition = -1;
        } else {
            adapterPosition = findContainingViewHolder.isRemoved() ? findContainingViewHolder.mOldPosition : findContainingViewHolder.getAdapterPosition();
        }
        a0Var.l = adapterPosition;
        this.mState.n = getDeepestFocusedViewWithId(findContainingViewHolder.itemView);
    }

    private void setAdapterInternal(g gVar, boolean z2, boolean z3) {
        g gVar2 = this.mAdapter;
        if (gVar2 != null) {
            gVar2.unregisterAdapterDataObserver(this.mObserver);
            this.mAdapter.onDetachedFromRecyclerView(this);
        }
        if (!z2 || z3) {
            removeAndRecycleViews();
        }
        b.w.b.a aVar = this.mAdapterHelper;
        aVar.l(aVar.f2703b);
        aVar.l(aVar.f2704c);
        aVar.f2707f = 0;
        g gVar3 = this.mAdapter;
        this.mAdapter = gVar;
        if (gVar != null) {
            gVar.registerAdapterDataObserver(this.mObserver);
            gVar.onAttachedToRecyclerView(this);
        }
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.onAdapterChanged(gVar3, this.mAdapter);
        }
        v vVar = this.mRecycler;
        g gVar4 = this.mAdapter;
        vVar.b();
        u d2 = vVar.d();
        Objects.requireNonNull(d2);
        if (gVar3 != null) {
            d2.f429b--;
        }
        if (!z2 && d2.f429b == 0) {
            for (int i2 = 0; i2 < d2.f428a.size(); i2++) {
                d2.f428a.valueAt(i2).f430a.clear();
            }
        }
        if (gVar4 != null) {
            d2.f429b++;
        }
        this.mState.f395f = true;
    }

    private void stopScrollersInternal() {
        this.mViewFlinger.c();
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.stopSmoothScroller();
        }
    }

    public void absorbGlows(int i2, int i3) {
        if (i2 < 0) {
            ensureLeftGlow();
            if (this.mLeftGlow.isFinished()) {
                this.mLeftGlow.onAbsorb(-i2);
            }
        } else if (i2 > 0) {
            ensureRightGlow();
            if (this.mRightGlow.isFinished()) {
                this.mRightGlow.onAbsorb(i2);
            }
        }
        if (i3 < 0) {
            ensureTopGlow();
            if (this.mTopGlow.isFinished()) {
                this.mTopGlow.onAbsorb(-i3);
            }
        } else if (i3 > 0) {
            ensureBottomGlow();
            if (this.mBottomGlow.isFinished()) {
                this.mBottomGlow.onAbsorb(i3);
            }
        }
        if (i2 == 0 && i3 == 0) {
            return;
        }
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        postInvalidateOnAnimation();
    }

    @Override // android.view.ViewGroup, android.view.View
    public void addFocusables(ArrayList<View> arrayList, int i2, int i3) {
        o oVar = this.mLayout;
        if (oVar == null || !oVar.onAddFocusables(this, arrayList, i2, i3)) {
            super.addFocusables(arrayList, i2, i3);
        }
    }

    public void addItemDecoration(n nVar, int i2) {
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.assertNotInLayoutOrScroll("Cannot add item decoration during a scroll  or layout");
        }
        if (this.mItemDecorations.isEmpty()) {
            setWillNotDraw(false);
        }
        if (i2 < 0) {
            this.mItemDecorations.add(nVar);
        } else {
            this.mItemDecorations.add(i2, nVar);
        }
        markItemDecorInsetsDirty();
        requestLayout();
    }

    public void addOnChildAttachStateChangeListener(q qVar) {
        if (this.mOnChildAttachStateListeners == null) {
            this.mOnChildAttachStateListeners = new ArrayList();
        }
        this.mOnChildAttachStateListeners.add(qVar);
    }

    public void addOnItemTouchListener(s sVar) {
        this.mOnItemTouchListeners.add(sVar);
    }

    public void addOnScrollListener(t tVar) {
        if (this.mScrollListeners == null) {
            this.mScrollListeners = new ArrayList();
        }
        this.mScrollListeners.add(tVar);
    }

    public void animateAppearance(d0 d0Var, l.c cVar, l.c cVar2) {
        boolean z2;
        int i2;
        int i3;
        d0Var.setIsRecyclable(false);
        b.w.b.w wVar = (b.w.b.w) this.mItemAnimator;
        Objects.requireNonNull(wVar);
        if (cVar != null && ((i2 = cVar.f415a) != (i3 = cVar2.f415a) || cVar.f416b != cVar2.f416b)) {
            z2 = wVar.i(d0Var, i2, cVar.f416b, i3, cVar2.f416b);
        } else {
            b.w.b.k kVar = (b.w.b.k) wVar;
            kVar.n(d0Var);
            d0Var.itemView.setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            kVar.j.add(d0Var);
            z2 = true;
        }
        if (z2) {
            postAnimationRunner();
        }
    }

    public void animateDisappearance(d0 d0Var, l.c cVar, l.c cVar2) {
        boolean z2;
        addAnimatingView(d0Var);
        d0Var.setIsRecyclable(false);
        b.w.b.w wVar = (b.w.b.w) this.mItemAnimator;
        Objects.requireNonNull(wVar);
        int i2 = cVar.f415a;
        int i3 = cVar.f416b;
        View view = d0Var.itemView;
        int left = cVar2 == null ? view.getLeft() : cVar2.f415a;
        int top = cVar2 == null ? view.getTop() : cVar2.f416b;
        if (!d0Var.isRemoved() && (i2 != left || i3 != top)) {
            view.layout(left, top, view.getWidth() + left, view.getHeight() + top);
            z2 = wVar.i(d0Var, i2, i3, left, top);
        } else {
            b.w.b.k kVar = (b.w.b.k) wVar;
            kVar.n(d0Var);
            kVar.i.add(d0Var);
            z2 = true;
        }
        if (z2) {
            postAnimationRunner();
        }
    }

    public void assertInLayoutOrScroll(String str) {
        if (isComputingLayout()) {
            return;
        }
        if (str == null) {
            throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x("Cannot call this method unless RecyclerView is computing a layout or scrolling")));
        }
        throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x(str)));
    }

    public void assertNotInLayoutOrScroll(String str) {
        if (isComputingLayout()) {
            if (str == null) {
                throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x("Cannot call this method while RecyclerView is computing a layout or scrolling")));
            }
            throw new IllegalStateException(str);
        } else if (this.mDispatchScrollCounter > 0) {
            Log.w(TAG, "Cannot call this method in a scroll callback. Scroll callbacks mightbe run during a measure & layout pass where you cannot change theRecyclerView data. Any method call that might change the structureof the RecyclerView or the adapter contents should be postponed tothe next frame.", new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x(""))));
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:20:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean canReuseUpdatedViewHolder(d0 d0Var) {
        boolean z2;
        l lVar = this.mItemAnimator;
        if (lVar != null) {
            List<Object> unmodifiedPayloads = d0Var.getUnmodifiedPayloads();
            b.w.b.k kVar = (b.w.b.k) lVar;
            Objects.requireNonNull(kVar);
            if (unmodifiedPayloads.isEmpty()) {
                if (!(!kVar.f2802g || d0Var.isInvalid())) {
                    z2 = false;
                    if (!z2) {
                        return false;
                    }
                }
            }
            z2 = true;
            if (!z2) {
            }
        }
        return true;
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return (layoutParams instanceof p) && this.mLayout.checkLayoutParams((p) layoutParams);
    }

    public void clearOldPositions() {
        int h2 = this.mChildHelper.h();
        for (int i2 = 0; i2 < h2; i2++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i2));
            if (!childViewHolderInt.shouldIgnore()) {
                childViewHolderInt.clearOldPosition();
            }
        }
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        for (int i3 = 0; i3 < size; i3++) {
            vVar.f436c.get(i3).clearOldPosition();
        }
        int size2 = vVar.f434a.size();
        for (int i4 = 0; i4 < size2; i4++) {
            vVar.f434a.get(i4).clearOldPosition();
        }
        ArrayList<d0> arrayList = vVar.f435b;
        if (arrayList != null) {
            int size3 = arrayList.size();
            for (int i5 = 0; i5 < size3; i5++) {
                vVar.f435b.get(i5).clearOldPosition();
            }
        }
    }

    public void clearOnChildAttachStateChangeListeners() {
        List<q> list = this.mOnChildAttachStateListeners;
        if (list != null) {
            list.clear();
        }
    }

    public void clearOnScrollListeners() {
        List<t> list = this.mScrollListeners;
        if (list != null) {
            list.clear();
        }
    }

    @Override // android.view.View
    public int computeHorizontalScrollExtent() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollHorizontally()) {
            return this.mLayout.computeHorizontalScrollExtent(this.mState);
        }
        return 0;
    }

    @Override // android.view.View
    public int computeHorizontalScrollOffset() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollHorizontally()) {
            return this.mLayout.computeHorizontalScrollOffset(this.mState);
        }
        return 0;
    }

    @Override // android.view.View
    public int computeHorizontalScrollRange() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollHorizontally()) {
            return this.mLayout.computeHorizontalScrollRange(this.mState);
        }
        return 0;
    }

    @Override // android.view.View
    public int computeVerticalScrollExtent() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollVertically()) {
            return this.mLayout.computeVerticalScrollExtent(this.mState);
        }
        return 0;
    }

    @Override // android.view.View
    public int computeVerticalScrollOffset() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollVertically()) {
            return this.mLayout.computeVerticalScrollOffset(this.mState);
        }
        return 0;
    }

    @Override // android.view.View
    public int computeVerticalScrollRange() {
        o oVar = this.mLayout;
        if (oVar != null && oVar.canScrollVertically()) {
            return this.mLayout.computeVerticalScrollRange(this.mState);
        }
        return 0;
    }

    public void considerReleasingGlowsOnScroll(int i2, int i3) {
        boolean z2;
        EdgeEffect edgeEffect = this.mLeftGlow;
        if (edgeEffect == null || edgeEffect.isFinished() || i2 <= 0) {
            z2 = false;
        } else {
            this.mLeftGlow.onRelease();
            z2 = this.mLeftGlow.isFinished();
        }
        EdgeEffect edgeEffect2 = this.mRightGlow;
        if (edgeEffect2 != null && !edgeEffect2.isFinished() && i2 < 0) {
            this.mRightGlow.onRelease();
            z2 |= this.mRightGlow.isFinished();
        }
        EdgeEffect edgeEffect3 = this.mTopGlow;
        if (edgeEffect3 != null && !edgeEffect3.isFinished() && i3 > 0) {
            this.mTopGlow.onRelease();
            z2 |= this.mTopGlow.isFinished();
        }
        EdgeEffect edgeEffect4 = this.mBottomGlow;
        if (edgeEffect4 != null && !edgeEffect4.isFinished() && i3 < 0) {
            this.mBottomGlow.onRelease();
            z2 |= this.mBottomGlow.isFinished();
        }
        if (z2) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            postInvalidateOnAnimation();
        }
    }

    public void consumePendingUpdateOperations() {
        if (this.mFirstLayoutComplete && !this.mDataSetHasChangedAfterLayout) {
            if (this.mAdapterHelper.g()) {
                b.w.b.a aVar = this.mAdapterHelper;
                int i2 = aVar.f2707f;
                if ((i2 & 4) != 0) {
                    if (!((i2 & 11) != 0)) {
                        int i3 = b.j.f.e.f2121a;
                        Trace.beginSection(TRACE_HANDLE_ADAPTER_UPDATES_TAG);
                        startInterceptRequestLayout();
                        onEnterLayoutOrScroll();
                        this.mAdapterHelper.j();
                        if (!this.mLayoutWasDefered) {
                            if (hasUpdatedView()) {
                                dispatchLayout();
                            } else {
                                this.mAdapterHelper.b();
                            }
                        }
                        stopInterceptRequestLayout(true);
                        onExitLayoutOrScroll();
                        Trace.endSection();
                        return;
                    }
                }
                if (aVar.g()) {
                    int i4 = b.j.f.e.f2121a;
                    Trace.beginSection(TRACE_ON_DATA_SET_CHANGE_LAYOUT_TAG);
                    dispatchLayout();
                    Trace.endSection();
                    return;
                }
                return;
            }
            return;
        }
        int i5 = b.j.f.e.f2121a;
        Trace.beginSection(TRACE_ON_DATA_SET_CHANGE_LAYOUT_TAG);
        dispatchLayout();
        Trace.endSection();
    }

    public void defaultOnMeasure(int i2, int i3) {
        int paddingRight = getPaddingRight() + getPaddingLeft();
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        setMeasuredDimension(o.chooseSize(i2, paddingRight, getMinimumWidth()), o.chooseSize(i3, getPaddingBottom() + getPaddingTop(), getMinimumHeight()));
    }

    public void dispatchChildAttached(View view) {
        d0 childViewHolderInt = getChildViewHolderInt(view);
        onChildAttachedToWindow(view);
        g gVar = this.mAdapter;
        if (gVar != null && childViewHolderInt != null) {
            gVar.onViewAttachedToWindow(childViewHolderInt);
        }
        List<q> list = this.mOnChildAttachStateListeners;
        if (list != null) {
            for (int size = list.size() - 1; size >= 0; size--) {
                this.mOnChildAttachStateListeners.get(size).b(view);
            }
        }
    }

    public void dispatchChildDetached(View view) {
        d0 childViewHolderInt = getChildViewHolderInt(view);
        onChildDetachedFromWindow(view);
        g gVar = this.mAdapter;
        if (gVar != null && childViewHolderInt != null) {
            gVar.onViewDetachedFromWindow(childViewHolderInt);
        }
        List<q> list = this.mOnChildAttachStateListeners;
        if (list != null) {
            for (int size = list.size() - 1; size >= 0; size--) {
                this.mOnChildAttachStateListeners.get(size).a(view);
            }
        }
    }

    public void dispatchLayout() {
        if (this.mAdapter == null) {
            Log.e(TAG, "No adapter attached; skipping layout");
        } else if (this.mLayout == null) {
            Log.e(TAG, "No layout manager attached; skipping layout");
        } else {
            a0 a0Var = this.mState;
            boolean z2 = false;
            a0Var.i = false;
            if (a0Var.f393d == 1) {
                dispatchLayoutStep1();
                this.mLayout.setExactMeasureSpecsFrom(this);
                dispatchLayoutStep2();
            } else {
                b.w.b.a aVar = this.mAdapterHelper;
                if (!aVar.f2704c.isEmpty() && !aVar.f2703b.isEmpty()) {
                    z2 = true;
                }
                if (!z2 && this.mLayout.getWidth() == getWidth() && this.mLayout.getHeight() == getHeight()) {
                    this.mLayout.setExactMeasureSpecsFrom(this);
                } else {
                    this.mLayout.setExactMeasureSpecsFrom(this);
                    dispatchLayoutStep2();
                }
            }
            dispatchLayoutStep3();
        }
    }

    @Override // android.view.View
    public boolean dispatchNestedFling(float f2, float f3, boolean z2) {
        return getScrollingChildHelper().a(f2, f3, z2);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreFling(float f2, float f3) {
        return getScrollingChildHelper().b(f2, f3);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreScroll(int i2, int i3, int[] iArr, int[] iArr2) {
        return getScrollingChildHelper().c(i2, i3, iArr, iArr2, 0);
    }

    @Override // android.view.View
    public boolean dispatchNestedScroll(int i2, int i3, int i4, int i5, int[] iArr) {
        return getScrollingChildHelper().d(i2, i3, i4, i5, iArr);
    }

    public void dispatchOnScrollStateChanged(int i2) {
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.onScrollStateChanged(i2);
        }
        onScrollStateChanged(i2);
        t tVar = this.mScrollListener;
        if (tVar != null) {
            tVar.onScrollStateChanged(this, i2);
        }
        List<t> list = this.mScrollListeners;
        if (list != null) {
            for (int size = list.size() - 1; size >= 0; size--) {
                this.mScrollListeners.get(size).onScrollStateChanged(this, i2);
            }
        }
    }

    public void dispatchOnScrolled(int i2, int i3) {
        this.mDispatchScrollCounter++;
        int scrollX = getScrollX();
        int scrollY = getScrollY();
        onScrollChanged(scrollX, scrollY, scrollX - i2, scrollY - i3);
        onScrolled(i2, i3);
        t tVar = this.mScrollListener;
        if (tVar != null) {
            tVar.onScrolled(this, i2, i3);
        }
        List<t> list = this.mScrollListeners;
        if (list != null) {
            for (int size = list.size() - 1; size >= 0; size--) {
                this.mScrollListeners.get(size).onScrolled(this, i2, i3);
            }
        }
        this.mDispatchScrollCounter--;
    }

    public void dispatchPendingImportantForAccessibilityChanges() {
        int i2;
        for (int size = this.mPendingAccessibilityImportanceChange.size() - 1; size >= 0; size--) {
            d0 d0Var = this.mPendingAccessibilityImportanceChange.get(size);
            if (d0Var.itemView.getParent() == this && !d0Var.shouldIgnore() && (i2 = d0Var.mPendingAccessibilityState) != -1) {
                View view = d0Var.itemView;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                view.setImportantForAccessibility(i2);
                d0Var.mPendingAccessibilityState = -1;
            }
        }
        this.mPendingAccessibilityImportanceChange.clear();
    }

    @Override // android.view.View
    public boolean dispatchPopulateAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        onPopulateAccessibilityEvent(accessibilityEvent);
        return true;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchRestoreInstanceState(SparseArray<Parcelable> sparseArray) {
        dispatchThawSelfOnly(sparseArray);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchSaveInstanceState(SparseArray<Parcelable> sparseArray) {
        dispatchFreezeSelfOnly(sparseArray);
    }

    @Override // android.view.View
    public void draw(Canvas canvas) {
        boolean z2;
        super.draw(canvas);
        int size = this.mItemDecorations.size();
        boolean z3 = false;
        for (int i2 = 0; i2 < size; i2++) {
            this.mItemDecorations.get(i2).onDrawOver(canvas, this, this.mState);
        }
        EdgeEffect edgeEffect = this.mLeftGlow;
        boolean z4 = true;
        if (edgeEffect == null || edgeEffect.isFinished()) {
            z2 = false;
        } else {
            int save = canvas.save();
            int paddingBottom = this.mClipToPadding ? getPaddingBottom() : 0;
            canvas.rotate(270.0f);
            canvas.translate((-getHeight()) + paddingBottom, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            EdgeEffect edgeEffect2 = this.mLeftGlow;
            z2 = edgeEffect2 != null && edgeEffect2.draw(canvas);
            canvas.restoreToCount(save);
        }
        EdgeEffect edgeEffect3 = this.mTopGlow;
        if (edgeEffect3 != null && !edgeEffect3.isFinished()) {
            int save2 = canvas.save();
            if (this.mClipToPadding) {
                canvas.translate(getPaddingLeft(), getPaddingTop());
            }
            EdgeEffect edgeEffect4 = this.mTopGlow;
            z2 |= edgeEffect4 != null && edgeEffect4.draw(canvas);
            canvas.restoreToCount(save2);
        }
        EdgeEffect edgeEffect5 = this.mRightGlow;
        if (edgeEffect5 != null && !edgeEffect5.isFinished()) {
            int save3 = canvas.save();
            int width = getWidth();
            int paddingTop = this.mClipToPadding ? getPaddingTop() : 0;
            canvas.rotate(90.0f);
            canvas.translate(-paddingTop, -width);
            EdgeEffect edgeEffect6 = this.mRightGlow;
            z2 |= edgeEffect6 != null && edgeEffect6.draw(canvas);
            canvas.restoreToCount(save3);
        }
        EdgeEffect edgeEffect7 = this.mBottomGlow;
        if (edgeEffect7 != null && !edgeEffect7.isFinished()) {
            int save4 = canvas.save();
            canvas.rotate(180.0f);
            if (this.mClipToPadding) {
                canvas.translate(getPaddingRight() + (-getWidth()), getPaddingBottom() + (-getHeight()));
            } else {
                canvas.translate(-getWidth(), -getHeight());
            }
            EdgeEffect edgeEffect8 = this.mBottomGlow;
            if (edgeEffect8 != null && edgeEffect8.draw(canvas)) {
                z3 = true;
            }
            z2 |= z3;
            canvas.restoreToCount(save4);
        }
        if (z2 || this.mItemAnimator == null || this.mItemDecorations.size() <= 0 || !this.mItemAnimator.g()) {
            z4 = z2;
        }
        if (z4) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            postInvalidateOnAnimation();
        }
    }

    @Override // android.view.ViewGroup
    public boolean drawChild(Canvas canvas, View view, long j2) {
        return super.drawChild(canvas, view, j2);
    }

    public void ensureBottomGlow() {
        if (this.mBottomGlow != null) {
            return;
        }
        EdgeEffect a2 = this.mEdgeEffectFactory.a(this);
        this.mBottomGlow = a2;
        if (this.mClipToPadding) {
            a2.setSize((getMeasuredWidth() - getPaddingLeft()) - getPaddingRight(), (getMeasuredHeight() - getPaddingTop()) - getPaddingBottom());
        } else {
            a2.setSize(getMeasuredWidth(), getMeasuredHeight());
        }
    }

    public void ensureLeftGlow() {
        if (this.mLeftGlow != null) {
            return;
        }
        EdgeEffect a2 = this.mEdgeEffectFactory.a(this);
        this.mLeftGlow = a2;
        if (this.mClipToPadding) {
            a2.setSize((getMeasuredHeight() - getPaddingTop()) - getPaddingBottom(), (getMeasuredWidth() - getPaddingLeft()) - getPaddingRight());
        } else {
            a2.setSize(getMeasuredHeight(), getMeasuredWidth());
        }
    }

    public void ensureRightGlow() {
        if (this.mRightGlow != null) {
            return;
        }
        EdgeEffect a2 = this.mEdgeEffectFactory.a(this);
        this.mRightGlow = a2;
        if (this.mClipToPadding) {
            a2.setSize((getMeasuredHeight() - getPaddingTop()) - getPaddingBottom(), (getMeasuredWidth() - getPaddingLeft()) - getPaddingRight());
        } else {
            a2.setSize(getMeasuredHeight(), getMeasuredWidth());
        }
    }

    public void ensureTopGlow() {
        if (this.mTopGlow != null) {
            return;
        }
        EdgeEffect a2 = this.mEdgeEffectFactory.a(this);
        this.mTopGlow = a2;
        if (this.mClipToPadding) {
            a2.setSize((getMeasuredWidth() - getPaddingLeft()) - getPaddingRight(), (getMeasuredHeight() - getPaddingTop()) - getPaddingBottom());
        } else {
            a2.setSize(getMeasuredWidth(), getMeasuredHeight());
        }
    }

    public String exceptionLabel() {
        StringBuilder x2 = c.b.a.a.a.x(" ");
        x2.append(super.toString());
        x2.append(", adapter:");
        x2.append(this.mAdapter);
        x2.append(", layout:");
        x2.append(this.mLayout);
        x2.append(", context:");
        x2.append(getContext());
        return x2.toString();
    }

    public final void fillRemainingScrollValues(a0 a0Var) {
        if (getScrollState() == 2) {
            OverScroller overScroller = this.mViewFlinger.f401d;
            overScroller.getFinalX();
            overScroller.getCurrX();
            Objects.requireNonNull(a0Var);
            overScroller.getFinalY();
            overScroller.getCurrY();
            return;
        }
        Objects.requireNonNull(a0Var);
    }

    public View findChildViewUnder(float f2, float f3) {
        for (int e2 = this.mChildHelper.e() - 1; e2 >= 0; e2--) {
            View d2 = this.mChildHelper.d(e2);
            float translationX = d2.getTranslationX();
            float translationY = d2.getTranslationY();
            if (f2 >= d2.getLeft() + translationX && f2 <= d2.getRight() + translationX && f3 >= d2.getTop() + translationY && f3 <= d2.getBottom() + translationY) {
                return d2;
            }
        }
        return null;
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:?, code lost:
        return r3;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public View findContainingItemView(View view) {
        ViewParent parent = view.getParent();
        while (parent != null && parent != this && (parent instanceof View)) {
            view = (View) parent;
            parent = view.getParent();
        }
        return null;
    }

    public d0 findContainingViewHolder(View view) {
        View findContainingItemView = findContainingItemView(view);
        if (findContainingItemView == null) {
            return null;
        }
        return getChildViewHolder(findContainingItemView);
    }

    public d0 findViewHolderForAdapterPosition(int i2) {
        d0 d0Var = null;
        if (this.mDataSetHasChangedAfterLayout) {
            return null;
        }
        int h2 = this.mChildHelper.h();
        for (int i3 = 0; i3 < h2; i3++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i3));
            if (childViewHolderInt != null && !childViewHolderInt.isRemoved() && getAdapterPositionFor(childViewHolderInt) == i2) {
                if (!this.mChildHelper.k(childViewHolderInt.itemView)) {
                    return childViewHolderInt;
                }
                d0Var = childViewHolderInt;
            }
        }
        return d0Var;
    }

    public d0 findViewHolderForItemId(long j2) {
        g gVar = this.mAdapter;
        d0 d0Var = null;
        if (gVar != null && gVar.hasStableIds()) {
            int h2 = this.mChildHelper.h();
            for (int i2 = 0; i2 < h2; i2++) {
                d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i2));
                if (childViewHolderInt != null && !childViewHolderInt.isRemoved() && childViewHolderInt.getItemId() == j2) {
                    if (!this.mChildHelper.k(childViewHolderInt.itemView)) {
                        return childViewHolderInt;
                    }
                    d0Var = childViewHolderInt;
                }
            }
        }
        return d0Var;
    }

    public d0 findViewHolderForLayoutPosition(int i2) {
        return findViewHolderForPosition(i2, false);
    }

    @Deprecated
    public d0 findViewHolderForPosition(int i2) {
        return findViewHolderForPosition(i2, false);
    }

    /* JADX WARN: Code restructure failed: missing block: B:112:0x015b, code lost:
        if (r0 < r13) goto L64;
     */
    /* JADX WARN: Removed duplicated region for block: B:116:0x0161  */
    /* JADX WARN: Removed duplicated region for block: B:117:0x0163  */
    /* JADX WARN: Removed duplicated region for block: B:119:0x016c  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean fling(int i2, int i3) {
        boolean z2;
        boolean z3;
        int i4;
        RecyclerView recyclerView;
        int i5;
        boolean z4;
        b.w.b.s f2;
        int position;
        PointF computeScrollVectorForPosition;
        boolean z5;
        o oVar = this.mLayout;
        boolean z6 = false;
        if (oVar == null) {
            Log.e(TAG, "Cannot fling without a LayoutManager set. Call setLayoutManager with a non-null argument.");
            return false;
        } else if (this.mLayoutSuppressed) {
            return false;
        } else {
            boolean canScrollHorizontally = oVar.canScrollHorizontally();
            boolean canScrollVertically = this.mLayout.canScrollVertically();
            int i6 = (!canScrollHorizontally || Math.abs(i2) < this.mMinFlingVelocity) ? 0 : i2;
            int i7 = (!canScrollVertically || Math.abs(i3) < this.mMinFlingVelocity) ? 0 : i3;
            if (i6 == 0 && i7 == 0) {
                return false;
            }
            float f3 = i6;
            float f4 = i7;
            if (dispatchNestedPreFling(f3, f4)) {
                return false;
            }
            boolean z7 = canScrollHorizontally || canScrollVertically;
            dispatchNestedFling(f3, f4, z7);
            r rVar = this.mOnFlingListener;
            if (rVar != null) {
                b.w.b.x xVar = (b.w.b.x) rVar;
                o layoutManager = xVar.f2803a.getLayoutManager();
                if (layoutManager == null || xVar.f2803a.getAdapter() == null) {
                    z2 = canScrollHorizontally;
                    z3 = canScrollVertically;
                } else {
                    int minFlingVelocity = xVar.f2803a.getMinFlingVelocity();
                    if (Math.abs(i7) > minFlingVelocity || Math.abs(i6) > minFlingVelocity) {
                        boolean z8 = layoutManager instanceof z.b;
                        if (z8) {
                            b.w.b.u uVar = (b.w.b.u) xVar;
                            b.w.b.t tVar = !z8 ? null : new b.w.b.t(uVar, uVar.f2803a.getContext());
                            if (tVar != null) {
                                int itemCount = layoutManager.getItemCount();
                                if (itemCount != 0) {
                                    if (layoutManager.canScrollVertically()) {
                                        f2 = uVar.g(layoutManager);
                                    } else {
                                        f2 = layoutManager.canScrollHorizontally() ? uVar.f(layoutManager) : null;
                                    }
                                    if (f2 != null) {
                                        int childCount = layoutManager.getChildCount();
                                        int i8 = Integer.MIN_VALUE;
                                        int i9 = Integer.MAX_VALUE;
                                        View view = null;
                                        z2 = canScrollHorizontally;
                                        int i10 = 0;
                                        View view2 = null;
                                        while (i10 < childCount) {
                                            int i11 = childCount;
                                            View childAt = layoutManager.getChildAt(i10);
                                            if (childAt == null) {
                                                z5 = canScrollVertically;
                                            } else {
                                                z5 = canScrollVertically;
                                                int d2 = uVar.d(childAt, f2);
                                                if (d2 <= 0 && d2 > i8) {
                                                    i8 = d2;
                                                    view = childAt;
                                                }
                                                if (d2 >= 0 && d2 < i9) {
                                                    i9 = d2;
                                                    view2 = childAt;
                                                }
                                            }
                                            i10++;
                                            childCount = i11;
                                            canScrollVertically = z5;
                                        }
                                        z3 = canScrollVertically;
                                        boolean z9 = !layoutManager.canScrollHorizontally() ? i7 <= 0 : i6 <= 0;
                                        if (z9 && view2 != null) {
                                            position = layoutManager.getPosition(view2);
                                        } else if (z9 || view == null) {
                                            if (z9) {
                                                view2 = view;
                                            }
                                            if (view2 != null) {
                                                position = ((z8 && (computeScrollVectorForPosition = ((z.b) layoutManager).computeScrollVectorForPosition(layoutManager.getItemCount() - 1)) != null && ((computeScrollVectorForPosition.x > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (computeScrollVectorForPosition.x == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1)) < 0 || (computeScrollVectorForPosition.y > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (computeScrollVectorForPosition.y == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1)) < 0)) == z9 ? -1 : 1) + layoutManager.getPosition(view2);
                                                if (position >= 0) {
                                                }
                                            }
                                            position = -1;
                                        } else {
                                            position = layoutManager.getPosition(view);
                                        }
                                        if (position != -1) {
                                            z6 = false;
                                        } else {
                                            tVar.setTargetPosition(position);
                                            layoutManager.startSmoothScroll(tVar);
                                            z6 = true;
                                        }
                                        if (z6) {
                                            z4 = true;
                                            z6 = z4;
                                        }
                                    }
                                }
                                z2 = canScrollHorizontally;
                                z3 = canScrollVertically;
                                position = -1;
                                if (position != -1) {
                                }
                                if (z6) {
                                }
                            }
                        }
                        z2 = canScrollHorizontally;
                        z3 = canScrollVertically;
                        if (z6) {
                        }
                    } else {
                        z2 = canScrollHorizontally;
                        z3 = canScrollVertically;
                    }
                    z4 = false;
                    z6 = z4;
                }
                i4 = 1;
                if (z6) {
                    return true;
                }
            } else {
                z2 = canScrollHorizontally;
                z3 = canScrollVertically;
                i4 = 1;
            }
            if (z7) {
                if (z3) {
                    i5 = z2 | 2;
                    recyclerView = this;
                } else {
                    recyclerView = this;
                    i5 = z2;
                }
                recyclerView.startNestedScroll(i5, i4);
                int i12 = recyclerView.mMaxFlingVelocity;
                int max = Math.max(-i12, Math.min(i6, i12));
                int i13 = recyclerView.mMaxFlingVelocity;
                int max2 = Math.max(-i13, Math.min(i7, i13));
                c0 c0Var = recyclerView.mViewFlinger;
                RecyclerView.this.setScrollState(2);
                c0Var.f400c = 0;
                c0Var.f399b = 0;
                Interpolator interpolator = c0Var.f402e;
                Interpolator interpolator2 = sQuinticInterpolator;
                if (interpolator != interpolator2) {
                    c0Var.f402e = interpolator2;
                    c0Var.f401d = new OverScroller(RecyclerView.this.getContext(), interpolator2);
                }
                c0Var.f401d.fling(0, 0, max, max2, Integer.MIN_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MAX_VALUE);
                c0Var.a();
                return true;
            }
            return false;
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public View focusSearch(View view, int i2) {
        View view2;
        boolean z2;
        View onInterceptFocusSearch = this.mLayout.onInterceptFocusSearch(view, i2);
        if (onInterceptFocusSearch != null) {
            return onInterceptFocusSearch;
        }
        boolean z3 = (this.mAdapter == null || this.mLayout == null || isComputingLayout() || this.mLayoutSuppressed) ? false : true;
        FocusFinder focusFinder = FocusFinder.getInstance();
        if (z3 && (i2 == 2 || i2 == 1)) {
            if (this.mLayout.canScrollVertically()) {
                int i3 = i2 == 2 ? 130 : 33;
                z2 = focusFinder.findNextFocus(this, view, i3) == null;
                if (FORCE_ABS_FOCUS_SEARCH_DIRECTION) {
                    i2 = i3;
                }
            } else {
                z2 = false;
            }
            if (!z2 && this.mLayout.canScrollHorizontally()) {
                int i4 = (this.mLayout.getLayoutDirection() == 1) ^ (i2 == 2) ? 66 : 17;
                boolean z4 = focusFinder.findNextFocus(this, view, i4) == null;
                if (FORCE_ABS_FOCUS_SEARCH_DIRECTION) {
                    i2 = i4;
                }
                z2 = z4;
            }
            if (z2) {
                consumePendingUpdateOperations();
                if (findContainingItemView(view) == null) {
                    return null;
                }
                startInterceptRequestLayout();
                this.mLayout.onFocusSearchFailed(view, i2, this.mRecycler, this.mState);
                stopInterceptRequestLayout(false);
            }
            view2 = focusFinder.findNextFocus(this, view, i2);
        } else {
            View findNextFocus = focusFinder.findNextFocus(this, view, i2);
            if (findNextFocus == null && z3) {
                consumePendingUpdateOperations();
                if (findContainingItemView(view) == null) {
                    return null;
                }
                startInterceptRequestLayout();
                view2 = this.mLayout.onFocusSearchFailed(view, i2, this.mRecycler, this.mState);
                stopInterceptRequestLayout(false);
            } else {
                view2 = findNextFocus;
            }
        }
        if (view2 == null || view2.hasFocusable()) {
            return isPreferredNextFocus(view, view2, i2) ? view2 : super.focusSearch(view, i2);
        } else if (getFocusedChild() == null) {
            return super.focusSearch(view, i2);
        } else {
            requestChildOnScreen(view2, null);
            return view;
        }
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateDefaultLayoutParams() {
        o oVar = this.mLayout;
        if (oVar != null) {
            return oVar.generateDefaultLayoutParams();
        }
        throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x("RecyclerView has no LayoutManager")));
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(AttributeSet attributeSet) {
        o oVar = this.mLayout;
        if (oVar != null) {
            return oVar.generateLayoutParams(getContext(), attributeSet);
        }
        throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x("RecyclerView has no LayoutManager")));
    }

    @Override // android.view.ViewGroup, android.view.View
    public CharSequence getAccessibilityClassName() {
        return "androidx.recyclerview.widget.RecyclerView";
    }

    public g getAdapter() {
        return this.mAdapter;
    }

    public int getAdapterPositionFor(d0 d0Var) {
        if (d0Var.hasAnyOfTheFlags(524) || !d0Var.isBound()) {
            return -1;
        }
        b.w.b.a aVar = this.mAdapterHelper;
        int i2 = d0Var.mPosition;
        int size = aVar.f2703b.size();
        for (int i3 = 0; i3 < size; i3++) {
            a.b bVar = aVar.f2703b.get(i3);
            int i4 = bVar.f2708a;
            if (i4 != 1) {
                if (i4 == 2) {
                    int i5 = bVar.f2709b;
                    if (i5 <= i2) {
                        int i6 = bVar.f2711d;
                        if (i5 + i6 > i2) {
                            return -1;
                        }
                        i2 -= i6;
                    } else {
                        continue;
                    }
                } else if (i4 == 8) {
                    int i7 = bVar.f2709b;
                    if (i7 == i2) {
                        i2 = bVar.f2711d;
                    } else {
                        if (i7 < i2) {
                            i2--;
                        }
                        if (bVar.f2711d <= i2) {
                            i2++;
                        }
                    }
                }
            } else if (bVar.f2709b <= i2) {
                i2 += bVar.f2711d;
            }
        }
        return i2;
    }

    @Override // android.view.View
    public int getBaseline() {
        o oVar = this.mLayout;
        if (oVar != null) {
            return oVar.getBaseline();
        }
        return super.getBaseline();
    }

    public long getChangedHolderKey(d0 d0Var) {
        return this.mAdapter.hasStableIds() ? d0Var.getItemId() : d0Var.mPosition;
    }

    public int getChildAdapterPosition(View view) {
        d0 childViewHolderInt = getChildViewHolderInt(view);
        if (childViewHolderInt != null) {
            return childViewHolderInt.getAdapterPosition();
        }
        return -1;
    }

    @Override // android.view.ViewGroup
    public int getChildDrawingOrder(int i2, int i3) {
        j jVar = this.mChildDrawingOrderCallback;
        if (jVar == null) {
            return super.getChildDrawingOrder(i2, i3);
        }
        return jVar.a(i2, i3);
    }

    public long getChildItemId(View view) {
        d0 childViewHolderInt;
        g gVar = this.mAdapter;
        if (gVar == null || !gVar.hasStableIds() || (childViewHolderInt = getChildViewHolderInt(view)) == null) {
            return -1L;
        }
        return childViewHolderInt.getItemId();
    }

    public int getChildLayoutPosition(View view) {
        d0 childViewHolderInt = getChildViewHolderInt(view);
        if (childViewHolderInt != null) {
            return childViewHolderInt.getLayoutPosition();
        }
        return -1;
    }

    @Deprecated
    public int getChildPosition(View view) {
        return getChildAdapterPosition(view);
    }

    public d0 getChildViewHolder(View view) {
        ViewParent parent = view.getParent();
        if (parent != null && parent != this) {
            throw new IllegalArgumentException("View " + view + " is not a direct child of " + this);
        }
        return getChildViewHolderInt(view);
    }

    @Override // android.view.ViewGroup
    public boolean getClipToPadding() {
        return this.mClipToPadding;
    }

    public b.w.b.v getCompatAccessibilityDelegate() {
        return this.mAccessibilityDelegate;
    }

    public void getDecoratedBoundsWithMargins(View view, Rect rect) {
        getDecoratedBoundsWithMarginsInt(view, rect);
    }

    public k getEdgeEffectFactory() {
        return this.mEdgeEffectFactory;
    }

    public l getItemAnimator() {
        return this.mItemAnimator;
    }

    public Rect getItemDecorInsetsForChild(View view) {
        p pVar = (p) view.getLayoutParams();
        if (!pVar.f426c) {
            return pVar.f425b;
        }
        if (this.mState.f396g && (pVar.b() || pVar.f424a.isInvalid())) {
            return pVar.f425b;
        }
        Rect rect = pVar.f425b;
        rect.set(0, 0, 0, 0);
        int size = this.mItemDecorations.size();
        for (int i2 = 0; i2 < size; i2++) {
            this.mTempRect.set(0, 0, 0, 0);
            this.mItemDecorations.get(i2).getItemOffsets(this.mTempRect, view, this, this.mState);
            int i3 = rect.left;
            Rect rect2 = this.mTempRect;
            rect.left = i3 + rect2.left;
            rect.top += rect2.top;
            rect.right += rect2.right;
            rect.bottom += rect2.bottom;
        }
        pVar.f426c = false;
        return rect;
    }

    public n getItemDecorationAt(int i2) {
        int itemDecorationCount = getItemDecorationCount();
        if (i2 >= 0 && i2 < itemDecorationCount) {
            return this.mItemDecorations.get(i2);
        }
        throw new IndexOutOfBoundsException(i2 + " is an invalid index for size " + itemDecorationCount);
    }

    public int getItemDecorationCount() {
        return this.mItemDecorations.size();
    }

    public o getLayoutManager() {
        return this.mLayout;
    }

    public int getMaxFlingVelocity() {
        return this.mMaxFlingVelocity;
    }

    public int getMinFlingVelocity() {
        return this.mMinFlingVelocity;
    }

    public long getNanoTime() {
        if (ALLOW_THREAD_GAP_WORK) {
            return System.nanoTime();
        }
        return 0L;
    }

    public r getOnFlingListener() {
        return this.mOnFlingListener;
    }

    public boolean getPreserveFocusAfterLayout() {
        return this.mPreserveFocusAfterLayout;
    }

    public u getRecycledViewPool() {
        return this.mRecycler.d();
    }

    public int getScrollState() {
        return this.mScrollState;
    }

    public boolean hasFixedSize() {
        return this.mHasFixedSize;
    }

    @Override // android.view.View
    public boolean hasNestedScrollingParent() {
        return getScrollingChildHelper().g(0);
    }

    public boolean hasPendingAdapterUpdates() {
        return !this.mFirstLayoutComplete || this.mDataSetHasChangedAfterLayout || this.mAdapterHelper.g();
    }

    public void initAdapterManager() {
        this.mAdapterHelper = new b.w.b.a(new f());
    }

    public void initFastScroller(StateListDrawable stateListDrawable, Drawable drawable, StateListDrawable stateListDrawable2, Drawable drawable2) {
        if (stateListDrawable != null && drawable != null && stateListDrawable2 != null && drawable2 != null) {
            Resources resources = getContext().getResources();
            new b.w.b.l(this, stateListDrawable, drawable, stateListDrawable2, drawable2, resources.getDimensionPixelSize(R.dimen.fastscroll_default_thickness), resources.getDimensionPixelSize(R.dimen.fastscroll_minimum_range), resources.getDimensionPixelOffset(R.dimen.fastscroll_margin));
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.i(this, c.b.a.a.a.x("Trying to set fast scroller without both required drawables.")));
    }

    public void invalidateGlows() {
        this.mBottomGlow = null;
        this.mTopGlow = null;
        this.mRightGlow = null;
        this.mLeftGlow = null;
    }

    public void invalidateItemDecorations() {
        if (this.mItemDecorations.size() == 0) {
            return;
        }
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.assertNotInLayoutOrScroll("Cannot invalidate item decorations during a scroll or layout");
        }
        markItemDecorInsetsDirty();
        requestLayout();
    }

    public boolean isAccessibilityEnabled() {
        AccessibilityManager accessibilityManager = this.mAccessibilityManager;
        return accessibilityManager != null && accessibilityManager.isEnabled();
    }

    public boolean isAnimating() {
        l lVar = this.mItemAnimator;
        return lVar != null && lVar.g();
    }

    @Override // android.view.View
    public boolean isAttachedToWindow() {
        return this.mIsAttached;
    }

    public boolean isComputingLayout() {
        return this.mLayoutOrScrollCounter > 0;
    }

    @Deprecated
    public boolean isLayoutFrozen() {
        return isLayoutSuppressed();
    }

    @Override // android.view.ViewGroup
    public final boolean isLayoutSuppressed() {
        return this.mLayoutSuppressed;
    }

    @Override // android.view.View
    public boolean isNestedScrollingEnabled() {
        return getScrollingChildHelper().f2207d;
    }

    public void jumpToPositionForSmoothScroller(int i2) {
        if (this.mLayout == null) {
            return;
        }
        setScrollState(2);
        this.mLayout.scrollToPosition(i2);
        awakenScrollBars();
    }

    public void markItemDecorInsetsDirty() {
        int h2 = this.mChildHelper.h();
        for (int i2 = 0; i2 < h2; i2++) {
            ((p) this.mChildHelper.g(i2).getLayoutParams()).f426c = true;
        }
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        for (int i3 = 0; i3 < size; i3++) {
            p pVar = (p) vVar.f436c.get(i3).itemView.getLayoutParams();
            if (pVar != null) {
                pVar.f426c = true;
            }
        }
    }

    public void markKnownViewsInvalid() {
        int h2 = this.mChildHelper.h();
        for (int i2 = 0; i2 < h2; i2++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i2));
            if (childViewHolderInt != null && !childViewHolderInt.shouldIgnore()) {
                childViewHolderInt.addFlags(6);
            }
        }
        markItemDecorInsetsDirty();
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        for (int i3 = 0; i3 < size; i3++) {
            d0 d0Var = vVar.f436c.get(i3);
            if (d0Var != null) {
                d0Var.addFlags(6);
                d0Var.addChangePayload(null);
            }
        }
        g gVar = RecyclerView.this.mAdapter;
        if (gVar == null || !gVar.hasStableIds()) {
            vVar.f();
        }
    }

    public void offsetChildrenHorizontal(int i2) {
        int e2 = this.mChildHelper.e();
        for (int i3 = 0; i3 < e2; i3++) {
            this.mChildHelper.d(i3).offsetLeftAndRight(i2);
        }
    }

    public void offsetChildrenVertical(int i2) {
        int e2 = this.mChildHelper.e();
        for (int i3 = 0; i3 < e2; i3++) {
            this.mChildHelper.d(i3).offsetTopAndBottom(i2);
        }
    }

    public void offsetPositionRecordsForInsert(int i2, int i3) {
        int h2 = this.mChildHelper.h();
        for (int i4 = 0; i4 < h2; i4++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i4));
            if (childViewHolderInt != null && !childViewHolderInt.shouldIgnore() && childViewHolderInt.mPosition >= i2) {
                childViewHolderInt.offsetPosition(i3, false);
                this.mState.f395f = true;
            }
        }
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        for (int i5 = 0; i5 < size; i5++) {
            d0 d0Var = vVar.f436c.get(i5);
            if (d0Var != null && d0Var.mPosition >= i2) {
                d0Var.offsetPosition(i3, true);
            }
        }
        requestLayout();
    }

    public void offsetPositionRecordsForMove(int i2, int i3) {
        int i4;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        int i10;
        int h2 = this.mChildHelper.h();
        int i11 = -1;
        if (i2 < i3) {
            i5 = i2;
            i4 = i3;
            i6 = -1;
        } else {
            i4 = i2;
            i5 = i3;
            i6 = 1;
        }
        for (int i12 = 0; i12 < h2; i12++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i12));
            if (childViewHolderInt != null && (i10 = childViewHolderInt.mPosition) >= i5 && i10 <= i4) {
                if (i10 == i2) {
                    childViewHolderInt.offsetPosition(i3 - i2, false);
                } else {
                    childViewHolderInt.offsetPosition(i6, false);
                }
                this.mState.f395f = true;
            }
        }
        v vVar = this.mRecycler;
        if (i2 < i3) {
            i8 = i2;
            i7 = i3;
        } else {
            i7 = i2;
            i11 = 1;
            i8 = i3;
        }
        int size = vVar.f436c.size();
        for (int i13 = 0; i13 < size; i13++) {
            d0 d0Var = vVar.f436c.get(i13);
            if (d0Var != null && (i9 = d0Var.mPosition) >= i8 && i9 <= i7) {
                if (i9 == i2) {
                    d0Var.offsetPosition(i3 - i2, false);
                } else {
                    d0Var.offsetPosition(i11, false);
                }
            }
        }
        requestLayout();
    }

    public void offsetPositionRecordsForRemove(int i2, int i3, boolean z2) {
        int i4 = i2 + i3;
        int h2 = this.mChildHelper.h();
        for (int i5 = 0; i5 < h2; i5++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i5));
            if (childViewHolderInt != null && !childViewHolderInt.shouldIgnore()) {
                int i6 = childViewHolderInt.mPosition;
                if (i6 >= i4) {
                    childViewHolderInt.offsetPosition(-i3, z2);
                    this.mState.f395f = true;
                } else if (i6 >= i2) {
                    childViewHolderInt.flagRemovedAndOffsetPosition(i2 - 1, -i3, z2);
                    this.mState.f395f = true;
                }
            }
        }
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        while (true) {
            size--;
            if (size >= 0) {
                d0 d0Var = vVar.f436c.get(size);
                if (d0Var != null) {
                    int i7 = d0Var.mPosition;
                    if (i7 >= i4) {
                        d0Var.offsetPosition(-i3, z2);
                    } else if (i7 >= i2) {
                        d0Var.addFlags(8);
                        vVar.g(size);
                    }
                }
            } else {
                requestLayout();
                return;
            }
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        this.mLayoutOrScrollCounter = 0;
        boolean z2 = true;
        this.mIsAttached = true;
        if (!this.mFirstLayoutComplete || isLayoutRequested()) {
            z2 = false;
        }
        this.mFirstLayoutComplete = z2;
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.dispatchAttachedToWindow(this);
        }
        this.mPostedAnimatorRunner = false;
        if (ALLOW_THREAD_GAP_WORK) {
            ThreadLocal<b.w.b.m> threadLocal = b.w.b.m.f2770b;
            b.w.b.m mVar = threadLocal.get();
            this.mGapWorker = mVar;
            if (mVar == null) {
                this.mGapWorker = new b.w.b.m();
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                Display display = getDisplay();
                float f2 = 60.0f;
                if (!isInEditMode() && display != null) {
                    float refreshRate = display.getRefreshRate();
                    if (refreshRate >= 30.0f) {
                        f2 = refreshRate;
                    }
                }
                b.w.b.m mVar2 = this.mGapWorker;
                mVar2.f2774f = 1.0E9f / f2;
                threadLocal.set(mVar2);
            }
            this.mGapWorker.f2772d.add(this);
        }
    }

    public void onChildAttachedToWindow(View view) {
    }

    public void onChildDetachedFromWindow(View view) {
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        b.w.b.m mVar;
        super.onDetachedFromWindow();
        l lVar = this.mItemAnimator;
        if (lVar != null) {
            lVar.f();
        }
        stopScroll();
        this.mIsAttached = false;
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.dispatchDetachedFromWindow(this, this.mRecycler);
        }
        this.mPendingAccessibilityImportanceChange.clear();
        removeCallbacks(this.mItemAnimatorRunner);
        Objects.requireNonNull(this.mViewInfoStore);
        do {
        } while (z.a.f2816a.b() != null);
        if (!ALLOW_THREAD_GAP_WORK || (mVar = this.mGapWorker) == null) {
            return;
        }
        mVar.f2772d.remove(this);
        this.mGapWorker = null;
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int size = this.mItemDecorations.size();
        for (int i2 = 0; i2 < size; i2++) {
            this.mItemDecorations.get(i2).onDraw(canvas, this, this.mState);
        }
    }

    public void onEnterLayoutOrScroll() {
        this.mLayoutOrScrollCounter++;
    }

    public void onExitLayoutOrScroll() {
        onExitLayoutOrScroll(true);
    }

    /* JADX WARN: Removed duplicated region for block: B:31:0x0068  */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onGenericMotionEvent(MotionEvent motionEvent) {
        float f2;
        float f3;
        if (this.mLayout != null && !this.mLayoutSuppressed && motionEvent.getAction() == 8) {
            if ((motionEvent.getSource() & 2) != 0) {
                f2 = this.mLayout.canScrollVertically() ? -motionEvent.getAxisValue(9) : 0.0f;
                if (this.mLayout.canScrollHorizontally()) {
                    f3 = motionEvent.getAxisValue(10);
                    if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || f3 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        scrollByInternal((int) (f3 * this.mScaledHorizontalScrollFactor), (int) (f2 * this.mScaledVerticalScrollFactor), motionEvent);
                    }
                }
                f3 = 0.0f;
                if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                }
                scrollByInternal((int) (f3 * this.mScaledHorizontalScrollFactor), (int) (f2 * this.mScaledVerticalScrollFactor), motionEvent);
            } else {
                if ((motionEvent.getSource() & Calib3d.CALIB_USE_EXTRINSIC_GUESS) != 0) {
                    float axisValue = motionEvent.getAxisValue(26);
                    if (!this.mLayout.canScrollVertically()) {
                        if (this.mLayout.canScrollHorizontally()) {
                            f3 = axisValue;
                            f2 = 0.0f;
                            if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            }
                            scrollByInternal((int) (f3 * this.mScaledHorizontalScrollFactor), (int) (f2 * this.mScaledVerticalScrollFactor), motionEvent);
                        }
                    } else {
                        f2 = -axisValue;
                        f3 = 0.0f;
                        if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        }
                        scrollByInternal((int) (f3 * this.mScaledHorizontalScrollFactor), (int) (f2 * this.mScaledVerticalScrollFactor), motionEvent);
                    }
                }
                f2 = 0.0f;
                f3 = 0.0f;
                if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                }
                scrollByInternal((int) (f3 * this.mScaledHorizontalScrollFactor), (int) (f2 * this.mScaledVerticalScrollFactor), motionEvent);
            }
        }
        return false;
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:41:0x00bd */
    @Override // android.view.ViewGroup
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        boolean z2;
        if (this.mLayoutSuppressed) {
            return false;
        }
        this.mInterceptingOnItemTouchListener = null;
        if (findInterceptingOnItemTouchListener(motionEvent)) {
            cancelScroll();
            return true;
        }
        o oVar = this.mLayout;
        if (oVar == null) {
            return false;
        }
        boolean canScrollHorizontally = oVar.canScrollHorizontally();
        boolean canScrollVertically = this.mLayout.canScrollVertically();
        if (this.mVelocityTracker == null) {
            this.mVelocityTracker = VelocityTracker.obtain();
        }
        this.mVelocityTracker.addMovement(motionEvent);
        int actionMasked = motionEvent.getActionMasked();
        int actionIndex = motionEvent.getActionIndex();
        if (actionMasked == 0) {
            if (this.mIgnoreMotionEventTillDown) {
                this.mIgnoreMotionEventTillDown = false;
            }
            this.mScrollPointerId = motionEvent.getPointerId(0);
            int x2 = (int) (motionEvent.getX() + 0.5f);
            this.mLastTouchX = x2;
            this.mInitialTouchX = x2;
            int y2 = (int) (motionEvent.getY() + 0.5f);
            this.mLastTouchY = y2;
            this.mInitialTouchY = y2;
            if (this.mScrollState == 2) {
                getParent().requestDisallowInterceptTouchEvent(true);
                setScrollState(1);
                stopNestedScroll(1);
            }
            int[] iArr = this.mNestedOffsets;
            iArr[1] = 0;
            iArr[0] = 0;
            int i2 = canScrollHorizontally;
            if (canScrollVertically) {
                i2 = (canScrollHorizontally ? 1 : 0) | 2;
            }
            startNestedScroll(i2, 0);
        } else if (actionMasked == 1) {
            this.mVelocityTracker.clear();
            stopNestedScroll(0);
        } else if (actionMasked == 2) {
            int findPointerIndex = motionEvent.findPointerIndex(this.mScrollPointerId);
            if (findPointerIndex < 0) {
                StringBuilder x3 = c.b.a.a.a.x("Error processing scroll; pointer index for id ");
                x3.append(this.mScrollPointerId);
                x3.append(" not found. Did any MotionEvents get skipped?");
                Log.e(TAG, x3.toString());
                return false;
            }
            int x4 = (int) (motionEvent.getX(findPointerIndex) + 0.5f);
            int y3 = (int) (motionEvent.getY(findPointerIndex) + 0.5f);
            if (this.mScrollState != 1) {
                int i3 = x4 - this.mInitialTouchX;
                int i4 = y3 - this.mInitialTouchY;
                if (!canScrollHorizontally || Math.abs(i3) <= this.mTouchSlop) {
                    z2 = false;
                } else {
                    this.mLastTouchX = x4;
                    z2 = true;
                }
                if (canScrollVertically && Math.abs(i4) > this.mTouchSlop) {
                    this.mLastTouchY = y3;
                    z2 = true;
                }
                if (z2) {
                    setScrollState(1);
                }
            }
        } else if (actionMasked == 3) {
            cancelScroll();
        } else if (actionMasked == 5) {
            this.mScrollPointerId = motionEvent.getPointerId(actionIndex);
            int x5 = (int) (motionEvent.getX(actionIndex) + 0.5f);
            this.mLastTouchX = x5;
            this.mInitialTouchX = x5;
            int y4 = (int) (motionEvent.getY(actionIndex) + 0.5f);
            this.mLastTouchY = y4;
            this.mInitialTouchY = y4;
        } else if (actionMasked == 6) {
            onPointerUp(motionEvent);
        }
        return this.mScrollState == 1;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z2, int i2, int i3, int i4, int i5) {
        int i6 = b.j.f.e.f2121a;
        Trace.beginSection(TRACE_ON_LAYOUT_TAG);
        dispatchLayout();
        Trace.endSection();
        this.mFirstLayoutComplete = true;
    }

    @Override // android.view.View
    public void onMeasure(int i2, int i3) {
        o oVar = this.mLayout;
        if (oVar == null) {
            defaultOnMeasure(i2, i3);
            return;
        }
        boolean z2 = false;
        if (oVar.isAutoMeasureEnabled()) {
            int mode = View.MeasureSpec.getMode(i2);
            int mode2 = View.MeasureSpec.getMode(i3);
            this.mLayout.onMeasure(this.mRecycler, this.mState, i2, i3);
            if (mode == 1073741824 && mode2 == 1073741824) {
                z2 = true;
            }
            if (z2 || this.mAdapter == null) {
                return;
            }
            if (this.mState.f393d == 1) {
                dispatchLayoutStep1();
            }
            this.mLayout.setMeasureSpecs(i2, i3);
            this.mState.i = true;
            dispatchLayoutStep2();
            this.mLayout.setMeasuredDimensionFromChildren(i2, i3);
            if (this.mLayout.shouldMeasureTwice()) {
                this.mLayout.setMeasureSpecs(View.MeasureSpec.makeMeasureSpec(getMeasuredWidth(), 1073741824), View.MeasureSpec.makeMeasureSpec(getMeasuredHeight(), 1073741824));
                this.mState.i = true;
                dispatchLayoutStep2();
                this.mLayout.setMeasuredDimensionFromChildren(i2, i3);
            }
        } else if (this.mHasFixedSize) {
            this.mLayout.onMeasure(this.mRecycler, this.mState, i2, i3);
        } else {
            if (this.mAdapterUpdateDuringMeasure) {
                startInterceptRequestLayout();
                onEnterLayoutOrScroll();
                processAdapterUpdatesAndSetAnimationFlags();
                onExitLayoutOrScroll();
                a0 a0Var = this.mState;
                if (a0Var.k) {
                    a0Var.f396g = true;
                } else {
                    this.mAdapterHelper.c();
                    this.mState.f396g = false;
                }
                this.mAdapterUpdateDuringMeasure = false;
                stopInterceptRequestLayout(false);
            } else if (this.mState.k) {
                setMeasuredDimension(getMeasuredWidth(), getMeasuredHeight());
                return;
            }
            g gVar = this.mAdapter;
            if (gVar != null) {
                this.mState.f394e = gVar.getItemCount();
            } else {
                this.mState.f394e = 0;
            }
            startInterceptRequestLayout();
            this.mLayout.onMeasure(this.mRecycler, this.mState, i2, i3);
            stopInterceptRequestLayout(false);
            this.mState.f396g = false;
        }
    }

    @Override // android.view.ViewGroup
    public boolean onRequestFocusInDescendants(int i2, Rect rect) {
        if (isComputingLayout()) {
            return false;
        }
        return super.onRequestFocusInDescendants(i2, rect);
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        Parcelable parcelable2;
        if (!(parcelable instanceof y)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        y yVar = (y) parcelable;
        this.mPendingSavedState = yVar;
        super.onRestoreInstanceState(yVar.getSuperState());
        o oVar = this.mLayout;
        if (oVar == null || (parcelable2 = this.mPendingSavedState.f443b) == null) {
            return;
        }
        oVar.onRestoreInstanceState(parcelable2);
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        y yVar = new y(super.onSaveInstanceState());
        y yVar2 = this.mPendingSavedState;
        if (yVar2 != null) {
            yVar.f443b = yVar2.f443b;
        } else {
            o oVar = this.mLayout;
            if (oVar != null) {
                yVar.f443b = oVar.onSaveInstanceState();
            } else {
                yVar.f443b = null;
            }
        }
        return yVar;
    }

    public void onScrollStateChanged(int i2) {
    }

    public void onScrolled(int i2, int i3) {
    }

    @Override // android.view.View
    public void onSizeChanged(int i2, int i3, int i4, int i5) {
        super.onSizeChanged(i2, i3, i4, i5);
        if (i2 == i4 && i3 == i5) {
            return;
        }
        invalidateGlows();
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:48:0x00dc */
    /* JADX WARN: Removed duplicated region for block: B:49:0x00de  */
    /* JADX WARN: Removed duplicated region for block: B:55:0x00f4  */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        boolean z2;
        boolean z3 = false;
        if (this.mLayoutSuppressed || this.mIgnoreMotionEventTillDown) {
            return false;
        }
        if (dispatchToOnItemTouchListeners(motionEvent)) {
            cancelScroll();
            return true;
        }
        o oVar = this.mLayout;
        if (oVar == null) {
            return false;
        }
        boolean canScrollHorizontally = oVar.canScrollHorizontally();
        boolean canScrollVertically = this.mLayout.canScrollVertically();
        if (this.mVelocityTracker == null) {
            this.mVelocityTracker = VelocityTracker.obtain();
        }
        int actionMasked = motionEvent.getActionMasked();
        int actionIndex = motionEvent.getActionIndex();
        if (actionMasked == 0) {
            int[] iArr = this.mNestedOffsets;
            iArr[1] = 0;
            iArr[0] = 0;
        }
        MotionEvent obtain = MotionEvent.obtain(motionEvent);
        int[] iArr2 = this.mNestedOffsets;
        obtain.offsetLocation(iArr2[0], iArr2[1]);
        if (actionMasked == 0) {
            this.mScrollPointerId = motionEvent.getPointerId(0);
            int x2 = (int) (motionEvent.getX() + 0.5f);
            this.mLastTouchX = x2;
            this.mInitialTouchX = x2;
            int y2 = (int) (motionEvent.getY() + 0.5f);
            this.mLastTouchY = y2;
            this.mInitialTouchY = y2;
            int i2 = canScrollHorizontally;
            if (canScrollVertically) {
                i2 = (canScrollHorizontally ? 1 : 0) | 2;
            }
            startNestedScroll(i2, 0);
        } else if (actionMasked == 1) {
            this.mVelocityTracker.addMovement(obtain);
            this.mVelocityTracker.computeCurrentVelocity(1000, this.mMaxFlingVelocity);
            float f2 = canScrollHorizontally ? -this.mVelocityTracker.getXVelocity(this.mScrollPointerId) : 0.0f;
            float f3 = canScrollVertically ? -this.mVelocityTracker.getYVelocity(this.mScrollPointerId) : 0.0f;
            if ((f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) || !fling((int) f2, (int) f3)) {
                setScrollState(0);
            }
            resetScroll();
            z3 = true;
        } else if (actionMasked == 2) {
            int findPointerIndex = motionEvent.findPointerIndex(this.mScrollPointerId);
            if (findPointerIndex < 0) {
                StringBuilder x3 = c.b.a.a.a.x("Error processing scroll; pointer index for id ");
                x3.append(this.mScrollPointerId);
                x3.append(" not found. Did any MotionEvents get skipped?");
                Log.e(TAG, x3.toString());
                return false;
            }
            int x4 = (int) (motionEvent.getX(findPointerIndex) + 0.5f);
            int y3 = (int) (motionEvent.getY(findPointerIndex) + 0.5f);
            int i3 = this.mLastTouchX - x4;
            int i4 = this.mLastTouchY - y3;
            if (this.mScrollState != 1) {
                if (canScrollHorizontally) {
                    if (i3 > 0) {
                        i3 = Math.max(0, i3 - this.mTouchSlop);
                    } else {
                        i3 = Math.min(0, i3 + this.mTouchSlop);
                    }
                    if (i3 != 0) {
                        z2 = true;
                        if (canScrollVertically) {
                            if (i4 > 0) {
                                i4 = Math.max(0, i4 - this.mTouchSlop);
                            } else {
                                i4 = Math.min(0, i4 + this.mTouchSlop);
                            }
                            if (i4 != 0) {
                                z2 = true;
                            }
                        }
                        if (z2) {
                            setScrollState(1);
                        }
                    }
                }
                z2 = false;
                if (canScrollVertically) {
                }
                if (z2) {
                }
            }
            int i5 = i3;
            int i6 = i4;
            if (this.mScrollState == 1) {
                int[] iArr3 = this.mReusableIntPair;
                iArr3[0] = 0;
                iArr3[1] = 0;
                if (dispatchNestedPreScroll(canScrollHorizontally ? i5 : 0, canScrollVertically ? i6 : 0, iArr3, this.mScrollOffset, 0)) {
                    int[] iArr4 = this.mReusableIntPair;
                    i5 -= iArr4[0];
                    i6 -= iArr4[1];
                    int[] iArr5 = this.mNestedOffsets;
                    int i7 = iArr5[0];
                    int[] iArr6 = this.mScrollOffset;
                    iArr5[0] = i7 + iArr6[0];
                    iArr5[1] = iArr5[1] + iArr6[1];
                    getParent().requestDisallowInterceptTouchEvent(true);
                }
                int i8 = i6;
                int[] iArr7 = this.mScrollOffset;
                this.mLastTouchX = x4 - iArr7[0];
                this.mLastTouchY = y3 - iArr7[1];
                if (scrollByInternal(canScrollHorizontally ? i5 : 0, canScrollVertically ? i8 : 0, motionEvent)) {
                    getParent().requestDisallowInterceptTouchEvent(true);
                }
                b.w.b.m mVar = this.mGapWorker;
                if (mVar != null && (i5 != 0 || i8 != 0)) {
                    mVar.a(this, i5, i8);
                }
            }
        } else if (actionMasked == 3) {
            cancelScroll();
        } else if (actionMasked == 5) {
            this.mScrollPointerId = motionEvent.getPointerId(actionIndex);
            int x5 = (int) (motionEvent.getX(actionIndex) + 0.5f);
            this.mLastTouchX = x5;
            this.mInitialTouchX = x5;
            int y4 = (int) (motionEvent.getY(actionIndex) + 0.5f);
            this.mLastTouchY = y4;
            this.mInitialTouchY = y4;
        } else if (actionMasked == 6) {
            onPointerUp(motionEvent);
        }
        if (!z3) {
            this.mVelocityTracker.addMovement(obtain);
        }
        obtain.recycle();
        return true;
    }

    public void postAnimationRunner() {
        if (this.mPostedAnimatorRunner || !this.mIsAttached) {
            return;
        }
        Runnable runnable = this.mItemAnimatorRunner;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        postOnAnimation(runnable);
        this.mPostedAnimatorRunner = true;
    }

    public void processDataSetCompletelyChanged(boolean z2) {
        this.mDispatchItemsChangedEvent = z2 | this.mDispatchItemsChangedEvent;
        this.mDataSetHasChangedAfterLayout = true;
        markKnownViewsInvalid();
    }

    public void recordAnimationInfoIfBouncedHiddenView(d0 d0Var, l.c cVar) {
        d0Var.setFlags(0, 8192);
        if (this.mState.f397h && d0Var.isUpdated() && !d0Var.isRemoved() && !d0Var.shouldIgnore()) {
            this.mViewInfoStore.f2815b.g(getChangedHolderKey(d0Var), d0Var);
        }
        this.mViewInfoStore.c(d0Var, cVar);
    }

    public void removeAndRecycleViews() {
        l lVar = this.mItemAnimator;
        if (lVar != null) {
            lVar.f();
        }
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.removeAndRecycleAllViews(this.mRecycler);
            this.mLayout.removeAndRecycleScrapInt(this.mRecycler);
        }
        this.mRecycler.b();
    }

    public boolean removeAnimatingView(View view) {
        startInterceptRequestLayout();
        b.w.b.b bVar = this.mChildHelper;
        int indexOfChild = RecyclerView.this.indexOfChild(view);
        boolean z2 = true;
        if (indexOfChild == -1) {
            bVar.m(view);
        } else if (bVar.f2713b.d(indexOfChild)) {
            bVar.f2713b.f(indexOfChild);
            bVar.m(view);
            ((e) bVar.f2712a).c(indexOfChild);
        } else {
            z2 = false;
        }
        if (z2) {
            d0 childViewHolderInt = getChildViewHolderInt(view);
            this.mRecycler.l(childViewHolderInt);
            this.mRecycler.i(childViewHolderInt);
        }
        stopInterceptRequestLayout(!z2);
        return z2;
    }

    @Override // android.view.ViewGroup
    public void removeDetachedView(View view, boolean z2) {
        d0 childViewHolderInt = getChildViewHolderInt(view);
        if (childViewHolderInt != null) {
            if (childViewHolderInt.isTmpDetached()) {
                childViewHolderInt.clearTmpDetachFlag();
            } else if (!childViewHolderInt.shouldIgnore()) {
                StringBuilder sb = new StringBuilder();
                sb.append("Called removeDetachedView with a view which is not flagged as tmp detached.");
                sb.append(childViewHolderInt);
                throw new IllegalArgumentException(c.b.a.a.a.i(this, sb));
            }
        }
        view.clearAnimation();
        dispatchChildDetached(view);
        super.removeDetachedView(view, z2);
    }

    public void removeItemDecoration(n nVar) {
        o oVar = this.mLayout;
        if (oVar != null) {
            oVar.assertNotInLayoutOrScroll("Cannot remove item decoration during a scroll  or layout");
        }
        this.mItemDecorations.remove(nVar);
        if (this.mItemDecorations.isEmpty()) {
            setWillNotDraw(getOverScrollMode() == 2);
        }
        markItemDecorInsetsDirty();
        requestLayout();
    }

    public void removeItemDecorationAt(int i2) {
        int itemDecorationCount = getItemDecorationCount();
        if (i2 >= 0 && i2 < itemDecorationCount) {
            removeItemDecoration(getItemDecorationAt(i2));
            return;
        }
        throw new IndexOutOfBoundsException(i2 + " is an invalid index for size " + itemDecorationCount);
    }

    public void removeOnChildAttachStateChangeListener(q qVar) {
        List<q> list = this.mOnChildAttachStateListeners;
        if (list == null) {
            return;
        }
        list.remove(qVar);
    }

    public void removeOnItemTouchListener(s sVar) {
        this.mOnItemTouchListeners.remove(sVar);
        if (this.mInterceptingOnItemTouchListener == sVar) {
            this.mInterceptingOnItemTouchListener = null;
        }
    }

    public void removeOnScrollListener(t tVar) {
        List<t> list = this.mScrollListeners;
        if (list != null) {
            list.remove(tVar);
        }
    }

    public void repositionShadowingViews() {
        d0 d0Var;
        int e2 = this.mChildHelper.e();
        for (int i2 = 0; i2 < e2; i2++) {
            View d2 = this.mChildHelper.d(i2);
            d0 childViewHolder = getChildViewHolder(d2);
            if (childViewHolder != null && (d0Var = childViewHolder.mShadowingHolder) != null) {
                View view = d0Var.itemView;
                int left = d2.getLeft();
                int top = d2.getTop();
                if (left != view.getLeft() || top != view.getTop()) {
                    view.layout(left, top, view.getWidth() + left, view.getHeight() + top);
                }
            }
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestChildFocus(View view, View view2) {
        if (!this.mLayout.onRequestChildFocus(this, this.mState, view, view2) && view2 != null) {
            requestChildOnScreen(view, view2);
        }
        super.requestChildFocus(view, view2);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean requestChildRectangleOnScreen(View view, Rect rect, boolean z2) {
        return this.mLayout.requestChildRectangleOnScreen(this, view, rect, z2);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestDisallowInterceptTouchEvent(boolean z2) {
        int size = this.mOnItemTouchListeners.size();
        for (int i2 = 0; i2 < size; i2++) {
            this.mOnItemTouchListeners.get(i2).c(z2);
        }
        super.requestDisallowInterceptTouchEvent(z2);
    }

    @Override // android.view.View, android.view.ViewParent
    public void requestLayout() {
        if (this.mInterceptRequestLayoutDepth == 0 && !this.mLayoutSuppressed) {
            super.requestLayout();
        } else {
            this.mLayoutWasDefered = true;
        }
    }

    public void saveOldPositions() {
        int h2 = this.mChildHelper.h();
        for (int i2 = 0; i2 < h2; i2++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i2));
            if (!childViewHolderInt.shouldIgnore()) {
                childViewHolderInt.saveOldPosition();
            }
        }
    }

    @Override // android.view.View
    public void scrollBy(int i2, int i3) {
        o oVar = this.mLayout;
        if (oVar == null) {
            Log.e(TAG, "Cannot scroll without a LayoutManager set. Call setLayoutManager with a non-null argument.");
        } else if (this.mLayoutSuppressed) {
        } else {
            boolean canScrollHorizontally = oVar.canScrollHorizontally();
            boolean canScrollVertically = this.mLayout.canScrollVertically();
            if (canScrollHorizontally || canScrollVertically) {
                if (!canScrollHorizontally) {
                    i2 = 0;
                }
                if (!canScrollVertically) {
                    i3 = 0;
                }
                scrollByInternal(i2, i3, null);
            }
        }
    }

    public boolean scrollByInternal(int i2, int i3, MotionEvent motionEvent) {
        int i4;
        int i5;
        int i6;
        int i7;
        consumePendingUpdateOperations();
        if (this.mAdapter != null) {
            int[] iArr = this.mReusableIntPair;
            iArr[0] = 0;
            iArr[1] = 0;
            scrollStep(i2, i3, iArr);
            int[] iArr2 = this.mReusableIntPair;
            int i8 = iArr2[0];
            int i9 = iArr2[1];
            i4 = i9;
            i5 = i8;
            i6 = i2 - i8;
            i7 = i3 - i9;
        } else {
            i4 = 0;
            i5 = 0;
            i6 = 0;
            i7 = 0;
        }
        if (!this.mItemDecorations.isEmpty()) {
            invalidate();
        }
        int[] iArr3 = this.mReusableIntPair;
        iArr3[0] = 0;
        iArr3[1] = 0;
        dispatchNestedScroll(i5, i4, i6, i7, this.mScrollOffset, 0, iArr3);
        int[] iArr4 = this.mReusableIntPair;
        int i10 = i6 - iArr4[0];
        int i11 = i7 - iArr4[1];
        boolean z2 = (iArr4[0] == 0 && iArr4[1] == 0) ? false : true;
        int i12 = this.mLastTouchX;
        int[] iArr5 = this.mScrollOffset;
        this.mLastTouchX = i12 - iArr5[0];
        this.mLastTouchY -= iArr5[1];
        int[] iArr6 = this.mNestedOffsets;
        iArr6[0] = iArr6[0] + iArr5[0];
        iArr6[1] = iArr6[1] + iArr5[1];
        if (getOverScrollMode() != 2) {
            if (motionEvent != null) {
                if (!((motionEvent.getSource() & 8194) == 8194)) {
                    pullGlows(motionEvent.getX(), i10, motionEvent.getY(), i11);
                }
            }
            considerReleasingGlowsOnScroll(i2, i3);
        }
        if (i5 != 0 || i4 != 0) {
            dispatchOnScrolled(i5, i4);
        }
        if (!awakenScrollBars()) {
            invalidate();
        }
        return (!z2 && i5 == 0 && i4 == 0) ? false : true;
    }

    public void scrollStep(int i2, int i3, int[] iArr) {
        startInterceptRequestLayout();
        onEnterLayoutOrScroll();
        int i4 = b.j.f.e.f2121a;
        Trace.beginSection(TRACE_SCROLL_TAG);
        fillRemainingScrollValues(this.mState);
        int scrollHorizontallyBy = i2 != 0 ? this.mLayout.scrollHorizontallyBy(i2, this.mRecycler, this.mState) : 0;
        int scrollVerticallyBy = i3 != 0 ? this.mLayout.scrollVerticallyBy(i3, this.mRecycler, this.mState) : 0;
        Trace.endSection();
        repositionShadowingViews();
        onExitLayoutOrScroll();
        stopInterceptRequestLayout(false);
        if (iArr != null) {
            iArr[0] = scrollHorizontallyBy;
            iArr[1] = scrollVerticallyBy;
        }
    }

    @Override // android.view.View
    public void scrollTo(int i2, int i3) {
        Log.w(TAG, "RecyclerView does not support scrolling to an absolute position. Use scrollToPosition instead");
    }

    public void scrollToPosition(int i2) {
        if (this.mLayoutSuppressed) {
            return;
        }
        stopScroll();
        o oVar = this.mLayout;
        if (oVar == null) {
            Log.e(TAG, "Cannot scroll to position a LayoutManager set. Call setLayoutManager with a non-null argument.");
            return;
        }
        oVar.scrollToPosition(i2);
        awakenScrollBars();
    }

    @Override // android.view.View, android.view.accessibility.AccessibilityEventSource
    public void sendAccessibilityEventUnchecked(AccessibilityEvent accessibilityEvent) {
        if (shouldDeferAccessibilityEvent(accessibilityEvent)) {
            return;
        }
        super.sendAccessibilityEventUnchecked(accessibilityEvent);
    }

    public void setAccessibilityDelegateCompat(b.w.b.v vVar) {
        this.mAccessibilityDelegate = vVar;
        b.j.j.q.n(this, vVar);
    }

    public void setAdapter(g gVar) {
        setLayoutFrozen(false);
        setAdapterInternal(gVar, false, true);
        processDataSetCompletelyChanged(false);
        requestLayout();
    }

    public void setChildDrawingOrderCallback(j jVar) {
        if (jVar == this.mChildDrawingOrderCallback) {
            return;
        }
        this.mChildDrawingOrderCallback = jVar;
        setChildrenDrawingOrderEnabled(jVar != null);
    }

    public boolean setChildImportantForAccessibilityInternal(d0 d0Var, int i2) {
        if (isComputingLayout()) {
            d0Var.mPendingAccessibilityState = i2;
            this.mPendingAccessibilityImportanceChange.add(d0Var);
            return false;
        }
        View view = d0Var.itemView;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        view.setImportantForAccessibility(i2);
        return true;
    }

    @Override // android.view.ViewGroup
    public void setClipToPadding(boolean z2) {
        if (z2 != this.mClipToPadding) {
            invalidateGlows();
        }
        this.mClipToPadding = z2;
        super.setClipToPadding(z2);
        if (this.mFirstLayoutComplete) {
            requestLayout();
        }
    }

    public void setEdgeEffectFactory(k kVar) {
        Objects.requireNonNull(kVar);
        this.mEdgeEffectFactory = kVar;
        invalidateGlows();
    }

    public void setHasFixedSize(boolean z2) {
        this.mHasFixedSize = z2;
    }

    public void setItemAnimator(l lVar) {
        l lVar2 = this.mItemAnimator;
        if (lVar2 != null) {
            lVar2.f();
            this.mItemAnimator.f409a = null;
        }
        this.mItemAnimator = lVar;
        if (lVar != null) {
            lVar.f409a = this.mItemAnimatorListener;
        }
    }

    public void setItemViewCacheSize(int i2) {
        v vVar = this.mRecycler;
        vVar.f438e = i2;
        vVar.m();
    }

    @Deprecated
    public void setLayoutFrozen(boolean z2) {
        suppressLayout(z2);
    }

    public void setLayoutManager(o oVar) {
        if (oVar == this.mLayout) {
            return;
        }
        stopScroll();
        if (this.mLayout != null) {
            l lVar = this.mItemAnimator;
            if (lVar != null) {
                lVar.f();
            }
            this.mLayout.removeAndRecycleAllViews(this.mRecycler);
            this.mLayout.removeAndRecycleScrapInt(this.mRecycler);
            this.mRecycler.b();
            if (this.mIsAttached) {
                this.mLayout.dispatchDetachedFromWindow(this, this.mRecycler);
            }
            this.mLayout.setRecyclerView(null);
            this.mLayout = null;
        } else {
            this.mRecycler.b();
        }
        b.w.b.b bVar = this.mChildHelper;
        b.a aVar = bVar.f2713b;
        aVar.f2715a = 0L;
        b.a aVar2 = aVar.f2716b;
        if (aVar2 != null) {
            aVar2.g();
        }
        int size = bVar.f2714c.size();
        while (true) {
            size--;
            if (size < 0) {
                break;
            }
            e eVar = (e) bVar.f2712a;
            Objects.requireNonNull(eVar);
            d0 childViewHolderInt = getChildViewHolderInt(bVar.f2714c.get(size));
            if (childViewHolderInt != null) {
                childViewHolderInt.onLeftHiddenState(RecyclerView.this);
            }
            bVar.f2714c.remove(size);
        }
        e eVar2 = (e) bVar.f2712a;
        int b2 = eVar2.b();
        for (int i2 = 0; i2 < b2; i2++) {
            View a2 = eVar2.a(i2);
            RecyclerView.this.dispatchChildDetached(a2);
            a2.clearAnimation();
        }
        RecyclerView.this.removeAllViews();
        this.mLayout = oVar;
        if (oVar != null) {
            if (oVar.mRecyclerView == null) {
                oVar.setRecyclerView(this);
                if (this.mIsAttached) {
                    this.mLayout.dispatchAttachedToWindow(this);
                }
            } else {
                StringBuilder sb = new StringBuilder();
                sb.append("LayoutManager ");
                sb.append(oVar);
                sb.append(" is already attached to a RecyclerView:");
                throw new IllegalArgumentException(c.b.a.a.a.i(oVar.mRecyclerView, sb));
            }
        }
        this.mRecycler.m();
        requestLayout();
    }

    @Override // android.view.ViewGroup
    @Deprecated
    public void setLayoutTransition(LayoutTransition layoutTransition) {
        if (layoutTransition == null) {
            super.setLayoutTransition(null);
            return;
        }
        throw new IllegalArgumentException("Providing a LayoutTransition into RecyclerView is not supported. Please use setItemAnimator() instead for animating changes to the items in this RecyclerView");
    }

    @Override // android.view.View
    public void setNestedScrollingEnabled(boolean z2) {
        b.j.j.f scrollingChildHelper = getScrollingChildHelper();
        if (scrollingChildHelper.f2207d) {
            View view = scrollingChildHelper.f2206c;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            view.stopNestedScroll();
        }
        scrollingChildHelper.f2207d = z2;
    }

    public void setOnFlingListener(r rVar) {
        this.mOnFlingListener = rVar;
    }

    @Deprecated
    public void setOnScrollListener(t tVar) {
        this.mScrollListener = tVar;
    }

    public void setPreserveFocusAfterLayout(boolean z2) {
        this.mPreserveFocusAfterLayout = z2;
    }

    public void setRecycledViewPool(u uVar) {
        u uVar2;
        v vVar = this.mRecycler;
        if (vVar.f440g != null) {
            uVar2.f429b--;
        }
        vVar.f440g = uVar;
        if (uVar == null || RecyclerView.this.getAdapter() == null) {
            return;
        }
        vVar.f440g.f429b++;
    }

    public void setRecyclerListener(w wVar) {
        this.mRecyclerListener = wVar;
    }

    public void setScrollState(int i2) {
        if (i2 == this.mScrollState) {
            return;
        }
        this.mScrollState = i2;
        if (i2 != 2) {
            stopScrollersInternal();
        }
        dispatchOnScrollStateChanged(i2);
    }

    public void setScrollingTouchSlop(int i2) {
        ViewConfiguration viewConfiguration = ViewConfiguration.get(getContext());
        if (i2 != 0) {
            if (i2 != 1) {
                Log.w(TAG, "setScrollingTouchSlop(): bad argument constant " + i2 + "; using default value");
            } else {
                this.mTouchSlop = viewConfiguration.getScaledPagingTouchSlop();
                return;
            }
        }
        this.mTouchSlop = viewConfiguration.getScaledTouchSlop();
    }

    public void setViewCacheExtension(b0 b0Var) {
        Objects.requireNonNull(this.mRecycler);
    }

    public boolean shouldDeferAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        if (isComputingLayout()) {
            int contentChangeTypes = accessibilityEvent != null ? accessibilityEvent.getContentChangeTypes() : 0;
            this.mEatenAccessibilityChangeFlags |= contentChangeTypes != 0 ? contentChangeTypes : 0;
            return true;
        }
        return false;
    }

    public void smoothScrollBy(int i2, int i3) {
        smoothScrollBy(i2, i3, null);
    }

    public void smoothScrollToPosition(int i2) {
        if (this.mLayoutSuppressed) {
            return;
        }
        o oVar = this.mLayout;
        if (oVar == null) {
            Log.e(TAG, "Cannot smooth scroll without a LayoutManager set. Call setLayoutManager with a non-null argument.");
        } else {
            oVar.smoothScrollToPosition(this, this.mState, i2);
        }
    }

    public void startInterceptRequestLayout() {
        int i2 = this.mInterceptRequestLayoutDepth + 1;
        this.mInterceptRequestLayoutDepth = i2;
        if (i2 != 1 || this.mLayoutSuppressed) {
            return;
        }
        this.mLayoutWasDefered = false;
    }

    @Override // android.view.View
    public boolean startNestedScroll(int i2) {
        return getScrollingChildHelper().h(i2, 0);
    }

    public void stopInterceptRequestLayout(boolean z2) {
        if (this.mInterceptRequestLayoutDepth < 1) {
            this.mInterceptRequestLayoutDepth = 1;
        }
        if (!z2 && !this.mLayoutSuppressed) {
            this.mLayoutWasDefered = false;
        }
        if (this.mInterceptRequestLayoutDepth == 1) {
            if (z2 && this.mLayoutWasDefered && !this.mLayoutSuppressed && this.mLayout != null && this.mAdapter != null) {
                dispatchLayout();
            }
            if (!this.mLayoutSuppressed) {
                this.mLayoutWasDefered = false;
            }
        }
        this.mInterceptRequestLayoutDepth--;
    }

    @Override // android.view.View
    public void stopNestedScroll() {
        getScrollingChildHelper().i(0);
    }

    public void stopScroll() {
        setScrollState(0);
        stopScrollersInternal();
    }

    @Override // android.view.ViewGroup
    public final void suppressLayout(boolean z2) {
        if (z2 != this.mLayoutSuppressed) {
            assertNotInLayoutOrScroll("Do not suppressLayout in layout or scroll");
            if (!z2) {
                this.mLayoutSuppressed = false;
                if (this.mLayoutWasDefered && this.mLayout != null && this.mAdapter != null) {
                    requestLayout();
                }
                this.mLayoutWasDefered = false;
                return;
            }
            long uptimeMillis = SystemClock.uptimeMillis();
            onTouchEvent(MotionEvent.obtain(uptimeMillis, uptimeMillis, 3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0));
            this.mLayoutSuppressed = true;
            this.mIgnoreMotionEventTillDown = true;
            stopScroll();
        }
    }

    public void swapAdapter(g gVar, boolean z2) {
        setLayoutFrozen(false);
        setAdapterInternal(gVar, true, z2);
        processDataSetCompletelyChanged(true);
        requestLayout();
    }

    public void viewRangeUpdate(int i2, int i3, Object obj) {
        int i4;
        int i5;
        int h2 = this.mChildHelper.h();
        int i6 = i3 + i2;
        for (int i7 = 0; i7 < h2; i7++) {
            View g2 = this.mChildHelper.g(i7);
            d0 childViewHolderInt = getChildViewHolderInt(g2);
            if (childViewHolderInt != null && !childViewHolderInt.shouldIgnore() && (i5 = childViewHolderInt.mPosition) >= i2 && i5 < i6) {
                childViewHolderInt.addFlags(2);
                childViewHolderInt.addChangePayload(obj);
                ((p) g2.getLayoutParams()).f426c = true;
            }
        }
        v vVar = this.mRecycler;
        int size = vVar.f436c.size();
        while (true) {
            size--;
            if (size < 0) {
                return;
            }
            d0 d0Var = vVar.f436c.get(size);
            if (d0Var != null && (i4 = d0Var.mPosition) >= i2 && i4 < i6) {
                d0Var.addFlags(2);
                vVar.g(size);
            }
        }
    }

    /* loaded from: classes.dex */
    public static abstract class g<VH extends d0> {
        private final h mObservable = new h();
        private boolean mHasStableIds = false;

        public final void bindViewHolder(VH vh, int i) {
            vh.mPosition = i;
            if (hasStableIds()) {
                vh.mItemId = getItemId(i);
            }
            vh.setFlags(1, 519);
            int i2 = b.j.f.e.f2121a;
            Trace.beginSection(RecyclerView.TRACE_BIND_VIEW_TAG);
            onBindViewHolder(vh, i, vh.getUnmodifiedPayloads());
            vh.clearPayload();
            ViewGroup.LayoutParams layoutParams = vh.itemView.getLayoutParams();
            if (layoutParams instanceof p) {
                ((p) layoutParams).f426c = true;
            }
            Trace.endSection();
        }

        public final VH createViewHolder(ViewGroup viewGroup, int i) {
            try {
                int i2 = b.j.f.e.f2121a;
                Trace.beginSection(RecyclerView.TRACE_CREATE_VIEW_TAG);
                VH onCreateViewHolder = onCreateViewHolder(viewGroup, i);
                if (onCreateViewHolder.itemView.getParent() == null) {
                    onCreateViewHolder.mItemViewType = i;
                    Trace.endSection();
                    return onCreateViewHolder;
                }
                throw new IllegalStateException("ViewHolder views must not be attached when created. Ensure that you are not passing 'true' to the attachToRoot parameter of LayoutInflater.inflate(..., boolean attachToRoot)");
            } catch (Throwable th) {
                int i3 = b.j.f.e.f2121a;
                Trace.endSection();
                throw th;
            }
        }

        public abstract int getItemCount();

        public long getItemId(int i) {
            return -1L;
        }

        public int getItemViewType(int i) {
            return 0;
        }

        public final boolean hasObservers() {
            return this.mObservable.a();
        }

        public final boolean hasStableIds() {
            return this.mHasStableIds;
        }

        public final void notifyDataSetChanged() {
            this.mObservable.b();
        }

        public final void notifyItemChanged(int i) {
            this.mObservable.d(i, 1, null);
        }

        public final void notifyItemInserted(int i) {
            this.mObservable.e(i, 1);
        }

        public final void notifyItemMoved(int i, int i2) {
            this.mObservable.c(i, i2);
        }

        public final void notifyItemRangeChanged(int i, int i2) {
            this.mObservable.d(i, i2, null);
        }

        public final void notifyItemRangeInserted(int i, int i2) {
            this.mObservable.e(i, i2);
        }

        public final void notifyItemRangeRemoved(int i, int i2) {
            this.mObservable.f(i, i2);
        }

        public final void notifyItemRemoved(int i) {
            this.mObservable.f(i, 1);
        }

        public void onAttachedToRecyclerView(RecyclerView recyclerView) {
        }

        public abstract void onBindViewHolder(VH vh, int i);

        public void onBindViewHolder(VH vh, int i, List<Object> list) {
            onBindViewHolder(vh, i);
        }

        public abstract VH onCreateViewHolder(ViewGroup viewGroup, int i);

        public void onDetachedFromRecyclerView(RecyclerView recyclerView) {
        }

        public boolean onFailedToRecycleView(VH vh) {
            return false;
        }

        public void onViewAttachedToWindow(VH vh) {
        }

        public void onViewDetachedFromWindow(VH vh) {
        }

        public void onViewRecycled(VH vh) {
        }

        public void registerAdapterDataObserver(i iVar) {
            this.mObservable.registerObserver(iVar);
        }

        public void setHasStableIds(boolean z) {
            if (!hasObservers()) {
                this.mHasStableIds = z;
                return;
            }
            throw new IllegalStateException("Cannot change whether this adapter has stable IDs while the adapter has registered observers.");
        }

        public void unregisterAdapterDataObserver(i iVar) {
            this.mObservable.unregisterObserver(iVar);
        }

        public final void notifyItemChanged(int i, Object obj) {
            this.mObservable.d(i, 1, obj);
        }

        public final void notifyItemRangeChanged(int i, int i2, Object obj) {
            this.mObservable.d(i, i2, obj);
        }
    }

    public RecyclerView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, R.attr.recyclerViewStyle);
    }

    public boolean dispatchNestedScroll(int i2, int i3, int i4, int i5, int[] iArr, int i6) {
        return getScrollingChildHelper().e(i2, i3, i4, i5, iArr, i6, null);
    }

    /* JADX WARN: Removed duplicated region for block: B:17:0x0034  */
    /* JADX WARN: Removed duplicated region for block: B:22:0x0036 A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public d0 findViewHolderForPosition(int i2, boolean z2) {
        int h2 = this.mChildHelper.h();
        d0 d0Var = null;
        for (int i3 = 0; i3 < h2; i3++) {
            d0 childViewHolderInt = getChildViewHolderInt(this.mChildHelper.g(i3));
            if (childViewHolderInt != null && !childViewHolderInt.isRemoved()) {
                if (z2) {
                    if (childViewHolderInt.mPosition != i2) {
                        continue;
                    }
                    if (this.mChildHelper.k(childViewHolderInt.itemView)) {
                        return childViewHolderInt;
                    }
                    d0Var = childViewHolderInt;
                } else {
                    if (childViewHolderInt.getLayoutPosition() != i2) {
                        continue;
                    }
                    if (this.mChildHelper.k(childViewHolderInt.itemView)) {
                    }
                }
            }
        }
        return d0Var;
    }

    public void onExitLayoutOrScroll(boolean z2) {
        int i2 = this.mLayoutOrScrollCounter - 1;
        this.mLayoutOrScrollCounter = i2;
        if (i2 < 1) {
            this.mLayoutOrScrollCounter = 0;
            if (z2) {
                dispatchContentChangedIfNecessary();
                dispatchPendingImportantForAccessibilityChanges();
            }
        }
    }

    public void smoothScrollBy(int i2, int i3, Interpolator interpolator) {
        smoothScrollBy(i2, i3, interpolator, Integer.MIN_VALUE);
    }

    public RecyclerView(Context context, AttributeSet attributeSet, int i2) {
        super(context, attributeSet, i2);
        float a2;
        float a3;
        TypedArray typedArray;
        this.mObserver = new x();
        this.mRecycler = new v();
        this.mViewInfoStore = new b.w.b.z();
        this.mUpdateChildViewsRunnable = new a();
        this.mTempRect = new Rect();
        this.mTempRect2 = new Rect();
        this.mTempRectF = new RectF();
        this.mItemDecorations = new ArrayList<>();
        this.mOnItemTouchListeners = new ArrayList<>();
        this.mInterceptRequestLayoutDepth = 0;
        this.mDataSetHasChangedAfterLayout = false;
        this.mDispatchItemsChangedEvent = false;
        this.mLayoutOrScrollCounter = 0;
        this.mDispatchScrollCounter = 0;
        this.mEdgeEffectFactory = new k();
        this.mItemAnimator = new b.w.b.k();
        this.mScrollState = 0;
        this.mScrollPointerId = -1;
        this.mScaledHorizontalScrollFactor = Float.MIN_VALUE;
        this.mScaledVerticalScrollFactor = Float.MIN_VALUE;
        this.mPreserveFocusAfterLayout = true;
        this.mViewFlinger = new c0();
        this.mPrefetchRegistry = ALLOW_THREAD_GAP_WORK ? new m.b() : null;
        this.mState = new a0();
        this.mItemsAddedOrRemoved = false;
        this.mItemsChanged = false;
        this.mItemAnimatorListener = new m();
        this.mPostedAnimatorRunner = false;
        this.mMinMaxLayoutPositions = new int[2];
        this.mScrollOffset = new int[2];
        this.mNestedOffsets = new int[2];
        this.mReusableIntPair = new int[2];
        this.mPendingAccessibilityImportanceChange = new ArrayList();
        this.mItemAnimatorRunner = new b();
        this.mViewInfoProcessCallback = new d();
        setScrollContainer(true);
        setFocusableInTouchMode(true);
        ViewConfiguration viewConfiguration = ViewConfiguration.get(context);
        this.mTouchSlop = viewConfiguration.getScaledTouchSlop();
        Method method = b.j.j.r.f2230a;
        int i3 = Build.VERSION.SDK_INT;
        if (i3 >= 26) {
            a2 = viewConfiguration.getScaledHorizontalScrollFactor();
        } else {
            a2 = b.j.j.r.a(viewConfiguration, context);
        }
        this.mScaledHorizontalScrollFactor = a2;
        if (i3 >= 26) {
            a3 = viewConfiguration.getScaledVerticalScrollFactor();
        } else {
            a3 = b.j.j.r.a(viewConfiguration, context);
        }
        this.mScaledVerticalScrollFactor = a3;
        this.mMinFlingVelocity = viewConfiguration.getScaledMinimumFlingVelocity();
        this.mMaxFlingVelocity = viewConfiguration.getScaledMaximumFlingVelocity();
        setWillNotDraw(getOverScrollMode() == 2);
        this.mItemAnimator.f409a = this.mItemAnimatorListener;
        initAdapterManager();
        initChildrenHelper();
        initAutofill();
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (getImportantForAccessibility() == 0) {
            setImportantForAccessibility(1);
        }
        this.mAccessibilityManager = (AccessibilityManager) getContext().getSystemService("accessibility");
        setAccessibilityDelegateCompat(new b.w.b.v(this));
        int[] iArr = b.w.a.f2701a;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, iArr, i2, 0);
        if (i3 >= 29) {
            typedArray = obtainStyledAttributes;
            saveAttributeDataForStyleable(context, iArr, attributeSet, obtainStyledAttributes, i2, 0);
        } else {
            typedArray = obtainStyledAttributes;
        }
        String string = typedArray.getString(8);
        if (typedArray.getInt(2, -1) == -1) {
            setDescendantFocusability(Calib3d.CALIB_TILTED_MODEL);
        }
        this.mClipToPadding = typedArray.getBoolean(1, true);
        boolean z2 = typedArray.getBoolean(3, false);
        this.mEnableFastScroller = z2;
        if (z2) {
            initFastScroller((StateListDrawable) typedArray.getDrawable(6), typedArray.getDrawable(7), (StateListDrawable) typedArray.getDrawable(4), typedArray.getDrawable(5));
        }
        typedArray.recycle();
        createLayoutManager(context, string, attributeSet, i2, 0);
        int[] iArr2 = NESTED_SCROLLING_ATTRS;
        TypedArray obtainStyledAttributes2 = context.obtainStyledAttributes(attributeSet, iArr2, i2, 0);
        if (i3 >= 29) {
            saveAttributeDataForStyleable(context, iArr2, attributeSet, obtainStyledAttributes2, i2, 0);
        }
        boolean z3 = obtainStyledAttributes2.getBoolean(0, true);
        obtainStyledAttributes2.recycle();
        setNestedScrollingEnabled(z3);
    }

    public boolean dispatchNestedPreScroll(int i2, int i3, int[] iArr, int[] iArr2, int i4) {
        return getScrollingChildHelper().c(i2, i3, iArr, iArr2, i4);
    }

    public boolean hasNestedScrollingParent(int i2) {
        return getScrollingChildHelper().f(i2) != null;
    }

    public void smoothScrollBy(int i2, int i3, Interpolator interpolator, int i4) {
        smoothScrollBy(i2, i3, interpolator, i4, false);
    }

    public boolean startNestedScroll(int i2, int i3) {
        return getScrollingChildHelper().h(i2, i3);
    }

    public void stopNestedScroll(int i2) {
        getScrollingChildHelper().i(i2);
    }

    /* loaded from: classes.dex */
    public static class p extends ViewGroup.MarginLayoutParams {

        /* renamed from: a  reason: collision with root package name */
        public d0 f424a;

        /* renamed from: b  reason: collision with root package name */
        public final Rect f425b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f426c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f427d;

        public p(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f425b = new Rect();
            this.f426c = true;
            this.f427d = false;
        }

        public int a() {
            return this.f424a.getLayoutPosition();
        }

        public boolean b() {
            return this.f424a.isUpdated();
        }

        public boolean c() {
            return this.f424a.isRemoved();
        }

        public p(int i, int i2) {
            super(i, i2);
            this.f425b = new Rect();
            this.f426c = true;
            this.f427d = false;
        }

        public p(ViewGroup.MarginLayoutParams marginLayoutParams) {
            super(marginLayoutParams);
            this.f425b = new Rect();
            this.f426c = true;
            this.f427d = false;
        }

        public p(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f425b = new Rect();
            this.f426c = true;
            this.f427d = false;
        }

        public p(p pVar) {
            super((ViewGroup.LayoutParams) pVar);
            this.f425b = new Rect();
            this.f426c = true;
            this.f427d = false;
        }
    }

    /* loaded from: classes.dex */
    public static class y extends b.l.a.a {
        public static final Parcelable.Creator<y> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public Parcelable f443b;

        /* loaded from: classes.dex */
        public static class a implements Parcelable.ClassLoaderCreator<y> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.ClassLoaderCreator
            public y createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new y(parcel, classLoader);
            }

            @Override // android.os.Parcelable.Creator
            public Object[] newArray(int i) {
                return new y[i];
            }

            @Override // android.os.Parcelable.Creator
            public Object createFromParcel(Parcel parcel) {
                return new y(parcel, null);
            }
        }

        public y(Parcel parcel, ClassLoader classLoader) {
            super(parcel, classLoader);
            this.f443b = parcel.readParcelable(classLoader == null ? o.class.getClassLoader() : classLoader);
        }

        @Override // b.l.a.a, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeParcelable(this.f443b, 0);
        }

        public y(Parcelable parcelable) {
            super(parcelable);
        }
    }

    public final void dispatchNestedScroll(int i2, int i3, int i4, int i5, int[] iArr, int i6, int[] iArr2) {
        getScrollingChildHelper().e(i2, i3, i4, i5, iArr, i6, iArr2);
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        o oVar = this.mLayout;
        if (oVar != null) {
            return oVar.generateLayoutParams(layoutParams);
        }
        throw new IllegalStateException(c.b.a.a.a.i(this, c.b.a.a.a.x("RecyclerView has no LayoutManager")));
    }

    public void smoothScrollBy(int i2, int i3, Interpolator interpolator, int i4, boolean z2) {
        o oVar = this.mLayout;
        if (oVar == null) {
            Log.e(TAG, "Cannot smooth scroll without a LayoutManager set. Call setLayoutManager with a non-null argument.");
        } else if (this.mLayoutSuppressed) {
        } else {
            if (!oVar.canScrollHorizontally()) {
                i2 = 0;
            }
            if (!this.mLayout.canScrollVertically()) {
                i3 = 0;
            }
            if (i2 == 0 && i3 == 0) {
                return;
            }
            if (i4 == Integer.MIN_VALUE || i4 > 0) {
                if (z2) {
                    int i5 = i2 != 0 ? 1 : 0;
                    if (i3 != 0) {
                        i5 |= 2;
                    }
                    startNestedScroll(i5, 1);
                }
                this.mViewFlinger.b(i2, i3, i4, interpolator);
                return;
            }
            scrollBy(i2, i3);
        }
    }

    public void addItemDecoration(n nVar) {
        addItemDecoration(nVar, -1);
    }
}