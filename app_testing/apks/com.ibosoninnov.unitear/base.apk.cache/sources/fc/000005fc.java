package b.z;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.TimeInterpolator;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.content.res.XmlResourceParser;
import android.graphics.Path;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.SparseArray;
import android.util.SparseIntArray;
import android.view.InflateException;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AnimationUtils;
import android.widget.ListView;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: Transition.java */
/* loaded from: classes.dex */
public abstract class j implements Cloneable {
    public static final boolean DBG = false;
    private static final String LOG_TAG = "Transition";
    private static final int MATCH_FIRST = 1;
    public static final int MATCH_ID = 3;
    private static final String MATCH_ID_STR = "id";
    public static final int MATCH_INSTANCE = 1;
    private static final String MATCH_INSTANCE_STR = "instance";
    public static final int MATCH_ITEM_ID = 4;
    private static final String MATCH_ITEM_ID_STR = "itemId";
    private static final int MATCH_LAST = 4;
    public static final int MATCH_NAME = 2;
    private static final String MATCH_NAME_STR = "name";
    private ArrayList<p> mEndValuesList;
    private e mEpicenterCallback;
    private b.f.a<String, String> mNameOverrides;
    public m mPropagation;
    private ArrayList<p> mStartValuesList;
    private static final int[] DEFAULT_MATCH_ORDER = {2, 1, 3, 4};
    private static final b.z.e STRAIGHT_PATH_MOTION = new a();
    private static ThreadLocal<b.f.a<Animator, d>> sRunningAnimators = new ThreadLocal<>();
    private String mName = getClass().getName();
    private long mStartDelay = -1;
    public long mDuration = -1;
    private TimeInterpolator mInterpolator = null;
    public ArrayList<Integer> mTargetIds = new ArrayList<>();
    public ArrayList<View> mTargets = new ArrayList<>();
    private ArrayList<String> mTargetNames = null;
    private ArrayList<Class<?>> mTargetTypes = null;
    private ArrayList<Integer> mTargetIdExcludes = null;
    private ArrayList<View> mTargetExcludes = null;
    private ArrayList<Class<?>> mTargetTypeExcludes = null;
    private ArrayList<String> mTargetNameExcludes = null;
    private ArrayList<Integer> mTargetIdChildExcludes = null;
    private ArrayList<View> mTargetChildExcludes = null;
    private ArrayList<Class<?>> mTargetTypeChildExcludes = null;
    private q mStartValues = new q();
    private q mEndValues = new q();
    public n mParent = null;
    private int[] mMatchOrder = DEFAULT_MATCH_ORDER;
    private ViewGroup mSceneRoot = null;
    public boolean mCanRemoveViews = false;
    public ArrayList<Animator> mCurrentAnimators = new ArrayList<>();
    private int mNumInstances = 0;
    private boolean mPaused = false;
    private boolean mEnded = false;
    private ArrayList<f> mListeners = null;
    private ArrayList<Animator> mAnimators = new ArrayList<>();
    private b.z.e mPathMotion = STRAIGHT_PATH_MOTION;

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public static class a extends b.z.e {
        @Override // b.z.e
        public Path getPath(float f2, float f3, float f4, float f5) {
            Path path = new Path();
            path.moveTo(f2, f3);
            path.lineTo(f4, f5);
            return path;
        }
    }

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public class b extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ b.f.a f2888a;

        public b(b.f.a aVar) {
            this.f2888a = aVar;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            this.f2888a.remove(animator);
            j.this.mCurrentAnimators.remove(animator);
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationStart(Animator animator) {
            j.this.mCurrentAnimators.add(animator);
        }
    }

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public class c extends AnimatorListenerAdapter {
        public c() {
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            j.this.end();
            animator.removeListener(this);
        }
    }

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public static class d {

        /* renamed from: a  reason: collision with root package name */
        public View f2891a;

        /* renamed from: b  reason: collision with root package name */
        public String f2892b;

        /* renamed from: c  reason: collision with root package name */
        public p f2893c;

        /* renamed from: d  reason: collision with root package name */
        public b0 f2894d;

        /* renamed from: e  reason: collision with root package name */
        public j f2895e;

        public d(View view, String str, j jVar, b0 b0Var, p pVar) {
            this.f2891a = view;
            this.f2892b = str;
            this.f2893c = pVar;
            this.f2894d = b0Var;
            this.f2895e = jVar;
        }
    }

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public static abstract class e {
        public abstract Rect a(j jVar);
    }

    /* compiled from: Transition.java */
    /* loaded from: classes.dex */
    public interface f {
        void onTransitionCancel(j jVar);

        void onTransitionEnd(j jVar);

        void onTransitionPause(j jVar);

        void onTransitionResume(j jVar);

        void onTransitionStart(j jVar);
    }

    public j() {
    }

    private void addUnmatched(b.f.a<View, p> aVar, b.f.a<View, p> aVar2) {
        for (int i = 0; i < aVar.f1775h; i++) {
            p l = aVar.l(i);
            if (isValidTarget(l.f2914b)) {
                this.mStartValuesList.add(l);
                this.mEndValuesList.add(null);
            }
        }
        for (int i2 = 0; i2 < aVar2.f1775h; i2++) {
            p l2 = aVar2.l(i2);
            if (isValidTarget(l2.f2914b)) {
                this.mEndValuesList.add(l2);
                this.mStartValuesList.add(null);
            }
        }
    }

    private static void addViewValues(q qVar, View view, p pVar) {
        qVar.f2916a.put(view, pVar);
        int id = view.getId();
        if (id >= 0) {
            if (qVar.f2917b.indexOfKey(id) >= 0) {
                qVar.f2917b.put(id, null);
            } else {
                qVar.f2917b.put(id, view);
            }
        }
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        String transitionName = view.getTransitionName();
        if (transitionName != null) {
            if (qVar.f2919d.e(transitionName) >= 0) {
                qVar.f2919d.put(transitionName, null);
            } else {
                qVar.f2919d.put(transitionName, view);
            }
        }
        if (view.getParent() instanceof ListView) {
            ListView listView = (ListView) view.getParent();
            if (listView.getAdapter().hasStableIds()) {
                long itemIdAtPosition = listView.getItemIdAtPosition(listView.getPositionForView(view));
                b.f.e<View> eVar = qVar.f2918c;
                if (eVar.f1750c) {
                    eVar.c();
                }
                if (b.f.d.b(eVar.f1751d, eVar.f1753f, itemIdAtPosition) >= 0) {
                    View d2 = qVar.f2918c.d(itemIdAtPosition);
                    if (d2 != null) {
                        d2.setHasTransientState(false);
                        qVar.f2918c.g(itemIdAtPosition, null);
                        return;
                    }
                    return;
                }
                view.setHasTransientState(true);
                qVar.f2918c.g(itemIdAtPosition, view);
            }
        }
    }

    private static boolean alreadyContains(int[] iArr, int i) {
        int i2 = iArr[i];
        for (int i3 = 0; i3 < i; i3++) {
            if (iArr[i3] == i2) {
                return true;
            }
        }
        return false;
    }

    private void captureHierarchy(View view, boolean z) {
        if (view == null) {
            return;
        }
        int id = view.getId();
        ArrayList<Integer> arrayList = this.mTargetIdExcludes;
        if (arrayList == null || !arrayList.contains(Integer.valueOf(id))) {
            ArrayList<View> arrayList2 = this.mTargetExcludes;
            if (arrayList2 == null || !arrayList2.contains(view)) {
                ArrayList<Class<?>> arrayList3 = this.mTargetTypeExcludes;
                if (arrayList3 != null) {
                    int size = arrayList3.size();
                    for (int i = 0; i < size; i++) {
                        if (this.mTargetTypeExcludes.get(i).isInstance(view)) {
                            return;
                        }
                    }
                }
                if (view.getParent() instanceof ViewGroup) {
                    p pVar = new p(view);
                    if (z) {
                        captureStartValues(pVar);
                    } else {
                        captureEndValues(pVar);
                    }
                    pVar.f2915c.add(this);
                    capturePropagationValues(pVar);
                    if (z) {
                        addViewValues(this.mStartValues, view, pVar);
                    } else {
                        addViewValues(this.mEndValues, view, pVar);
                    }
                }
                if (view instanceof ViewGroup) {
                    ArrayList<Integer> arrayList4 = this.mTargetIdChildExcludes;
                    if (arrayList4 == null || !arrayList4.contains(Integer.valueOf(id))) {
                        ArrayList<View> arrayList5 = this.mTargetChildExcludes;
                        if (arrayList5 == null || !arrayList5.contains(view)) {
                            ArrayList<Class<?>> arrayList6 = this.mTargetTypeChildExcludes;
                            if (arrayList6 != null) {
                                int size2 = arrayList6.size();
                                for (int i2 = 0; i2 < size2; i2++) {
                                    if (this.mTargetTypeChildExcludes.get(i2).isInstance(view)) {
                                        return;
                                    }
                                }
                            }
                            ViewGroup viewGroup = (ViewGroup) view;
                            for (int i3 = 0; i3 < viewGroup.getChildCount(); i3++) {
                                captureHierarchy(viewGroup.getChildAt(i3), z);
                            }
                        }
                    }
                }
            }
        }
    }

    private ArrayList<Integer> excludeId(ArrayList<Integer> arrayList, int i, boolean z) {
        if (i > 0) {
            if (z) {
                return b.v.u.c.c(arrayList, Integer.valueOf(i));
            }
            return b.v.u.c.x(arrayList, Integer.valueOf(i));
        }
        return arrayList;
    }

    private static <T> ArrayList<T> excludeObject(ArrayList<T> arrayList, T t, boolean z) {
        if (t != null) {
            if (z) {
                return b.v.u.c.c(arrayList, t);
            }
            return b.v.u.c.x(arrayList, t);
        }
        return arrayList;
    }

    private ArrayList<Class<?>> excludeType(ArrayList<Class<?>> arrayList, Class<?> cls, boolean z) {
        if (cls != null) {
            if (z) {
                return b.v.u.c.c(arrayList, cls);
            }
            return b.v.u.c.x(arrayList, cls);
        }
        return arrayList;
    }

    private ArrayList<View> excludeView(ArrayList<View> arrayList, View view, boolean z) {
        if (view != null) {
            if (z) {
                return b.v.u.c.c(arrayList, view);
            }
            return b.v.u.c.x(arrayList, view);
        }
        return arrayList;
    }

    private static b.f.a<Animator, d> getRunningAnimators() {
        b.f.a<Animator, d> aVar = sRunningAnimators.get();
        if (aVar == null) {
            b.f.a<Animator, d> aVar2 = new b.f.a<>();
            sRunningAnimators.set(aVar2);
            return aVar2;
        }
        return aVar;
    }

    private static boolean isValidMatch(int i) {
        return i >= 1 && i <= 4;
    }

    private static boolean isValueChanged(p pVar, p pVar2, String str) {
        Object obj = pVar.f2913a.get(str);
        Object obj2 = pVar2.f2913a.get(str);
        if (obj == null && obj2 == null) {
            return false;
        }
        if (obj == null || obj2 == null) {
            return true;
        }
        return true ^ obj.equals(obj2);
    }

    private void matchIds(b.f.a<View, p> aVar, b.f.a<View, p> aVar2, SparseArray<View> sparseArray, SparseArray<View> sparseArray2) {
        View view;
        int size = sparseArray.size();
        for (int i = 0; i < size; i++) {
            View valueAt = sparseArray.valueAt(i);
            if (valueAt != null && isValidTarget(valueAt) && (view = sparseArray2.get(sparseArray.keyAt(i))) != null && isValidTarget(view)) {
                p orDefault = aVar.getOrDefault(valueAt, null);
                p orDefault2 = aVar2.getOrDefault(view, null);
                if (orDefault != null && orDefault2 != null) {
                    this.mStartValuesList.add(orDefault);
                    this.mEndValuesList.add(orDefault2);
                    aVar.remove(valueAt);
                    aVar2.remove(view);
                }
            }
        }
    }

    private void matchInstances(b.f.a<View, p> aVar, b.f.a<View, p> aVar2) {
        p remove;
        for (int i = aVar.f1775h - 1; i >= 0; i--) {
            View h2 = aVar.h(i);
            if (h2 != null && isValidTarget(h2) && (remove = aVar2.remove(h2)) != null && isValidTarget(remove.f2914b)) {
                this.mStartValuesList.add(aVar.j(i));
                this.mEndValuesList.add(remove);
            }
        }
    }

    private void matchItemIds(b.f.a<View, p> aVar, b.f.a<View, p> aVar2, b.f.e<View> eVar, b.f.e<View> eVar2) {
        View d2;
        int h2 = eVar.h();
        for (int i = 0; i < h2; i++) {
            View i2 = eVar.i(i);
            if (i2 != null && isValidTarget(i2) && (d2 = eVar2.d(eVar.f(i))) != null && isValidTarget(d2)) {
                p orDefault = aVar.getOrDefault(i2, null);
                p orDefault2 = aVar2.getOrDefault(d2, null);
                if (orDefault != null && orDefault2 != null) {
                    this.mStartValuesList.add(orDefault);
                    this.mEndValuesList.add(orDefault2);
                    aVar.remove(i2);
                    aVar2.remove(d2);
                }
            }
        }
    }

    private void matchNames(b.f.a<View, p> aVar, b.f.a<View, p> aVar2, b.f.a<String, View> aVar3, b.f.a<String, View> aVar4) {
        View view;
        int i = aVar3.f1775h;
        for (int i2 = 0; i2 < i; i2++) {
            View l = aVar3.l(i2);
            if (l != null && isValidTarget(l) && (view = aVar4.get(aVar3.h(i2))) != null && isValidTarget(view)) {
                p orDefault = aVar.getOrDefault(l, null);
                p orDefault2 = aVar2.getOrDefault(view, null);
                if (orDefault != null && orDefault2 != null) {
                    this.mStartValuesList.add(orDefault);
                    this.mEndValuesList.add(orDefault2);
                    aVar.remove(l);
                    aVar2.remove(view);
                }
            }
        }
    }

    private void matchStartAndEnd(q qVar, q qVar2) {
        b.f.a<View, p> aVar = new b.f.a<>(qVar.f2916a);
        b.f.a<View, p> aVar2 = new b.f.a<>(qVar2.f2916a);
        int i = 0;
        while (true) {
            int[] iArr = this.mMatchOrder;
            if (i < iArr.length) {
                int i2 = iArr[i];
                if (i2 == 1) {
                    matchInstances(aVar, aVar2);
                } else if (i2 == 2) {
                    matchNames(aVar, aVar2, qVar.f2919d, qVar2.f2919d);
                } else if (i2 == 3) {
                    matchIds(aVar, aVar2, qVar.f2917b, qVar2.f2917b);
                } else if (i2 == 4) {
                    matchItemIds(aVar, aVar2, qVar.f2918c, qVar2.f2918c);
                }
                i++;
            } else {
                addUnmatched(aVar, aVar2);
                return;
            }
        }
    }

    private static int[] parseMatchOrder(String str) {
        StringTokenizer stringTokenizer = new StringTokenizer(str, ",");
        int[] iArr = new int[stringTokenizer.countTokens()];
        int i = 0;
        while (stringTokenizer.hasMoreTokens()) {
            String trim = stringTokenizer.nextToken().trim();
            if (MATCH_ID_STR.equalsIgnoreCase(trim)) {
                iArr[i] = 3;
            } else if ("instance".equalsIgnoreCase(trim)) {
                iArr[i] = 1;
            } else if ("name".equalsIgnoreCase(trim)) {
                iArr[i] = 2;
            } else if (MATCH_ITEM_ID_STR.equalsIgnoreCase(trim)) {
                iArr[i] = 4;
            } else if (trim.isEmpty()) {
                int[] iArr2 = new int[iArr.length - 1];
                System.arraycopy(iArr, 0, iArr2, 0, i);
                i--;
                iArr = iArr2;
            } else {
                throw new InflateException(c.b.a.a.a.r("Unknown match type in matchOrder: '", trim, "'"));
            }
            i++;
        }
        return iArr;
    }

    private void runAnimator(Animator animator, b.f.a<Animator, d> aVar) {
        if (animator != null) {
            animator.addListener(new b(aVar));
            animate(animator);
        }
    }

    public j addListener(f fVar) {
        if (this.mListeners == null) {
            this.mListeners = new ArrayList<>();
        }
        this.mListeners.add(fVar);
        return this;
    }

    public j addTarget(View view) {
        this.mTargets.add(view);
        return this;
    }

    public void animate(Animator animator) {
        if (animator == null) {
            end();
            return;
        }
        if (getDuration() >= 0) {
            animator.setDuration(getDuration());
        }
        if (getStartDelay() >= 0) {
            animator.setStartDelay(animator.getStartDelay() + getStartDelay());
        }
        if (getInterpolator() != null) {
            animator.setInterpolator(getInterpolator());
        }
        animator.addListener(new c());
        animator.start();
    }

    public void cancel() {
        for (int size = this.mCurrentAnimators.size() - 1; size >= 0; size--) {
            this.mCurrentAnimators.get(size).cancel();
        }
        ArrayList<f> arrayList = this.mListeners;
        if (arrayList == null || arrayList.size() <= 0) {
            return;
        }
        ArrayList arrayList2 = (ArrayList) this.mListeners.clone();
        int size2 = arrayList2.size();
        for (int i = 0; i < size2; i++) {
            ((f) arrayList2.get(i)).onTransitionCancel(this);
        }
    }

    public abstract void captureEndValues(p pVar);

    public void capturePropagationValues(p pVar) {
        if (this.mPropagation != null && !pVar.f2913a.isEmpty()) {
            throw null;
        }
    }

    public abstract void captureStartValues(p pVar);

    public void captureValues(ViewGroup viewGroup, boolean z) {
        ArrayList<String> arrayList;
        ArrayList<Class<?>> arrayList2;
        b.f.a<String, String> aVar;
        clearValues(z);
        if ((this.mTargetIds.size() <= 0 && this.mTargets.size() <= 0) || (((arrayList = this.mTargetNames) != null && !arrayList.isEmpty()) || ((arrayList2 = this.mTargetTypes) != null && !arrayList2.isEmpty()))) {
            captureHierarchy(viewGroup, z);
        } else {
            for (int i = 0; i < this.mTargetIds.size(); i++) {
                View findViewById = viewGroup.findViewById(this.mTargetIds.get(i).intValue());
                if (findViewById != null) {
                    p pVar = new p(findViewById);
                    if (z) {
                        captureStartValues(pVar);
                    } else {
                        captureEndValues(pVar);
                    }
                    pVar.f2915c.add(this);
                    capturePropagationValues(pVar);
                    if (z) {
                        addViewValues(this.mStartValues, findViewById, pVar);
                    } else {
                        addViewValues(this.mEndValues, findViewById, pVar);
                    }
                }
            }
            for (int i2 = 0; i2 < this.mTargets.size(); i2++) {
                View view = this.mTargets.get(i2);
                p pVar2 = new p(view);
                if (z) {
                    captureStartValues(pVar2);
                } else {
                    captureEndValues(pVar2);
                }
                pVar2.f2915c.add(this);
                capturePropagationValues(pVar2);
                if (z) {
                    addViewValues(this.mStartValues, view, pVar2);
                } else {
                    addViewValues(this.mEndValues, view, pVar2);
                }
            }
        }
        if (z || (aVar = this.mNameOverrides) == null) {
            return;
        }
        int i3 = aVar.f1775h;
        ArrayList arrayList3 = new ArrayList(i3);
        for (int i4 = 0; i4 < i3; i4++) {
            arrayList3.add(this.mStartValues.f2919d.remove(this.mNameOverrides.h(i4)));
        }
        for (int i5 = 0; i5 < i3; i5++) {
            View view2 = (View) arrayList3.get(i5);
            if (view2 != null) {
                this.mStartValues.f2919d.put(this.mNameOverrides.l(i5), view2);
            }
        }
    }

    public void clearValues(boolean z) {
        if (z) {
            this.mStartValues.f2916a.clear();
            this.mStartValues.f2917b.clear();
            this.mStartValues.f2918c.a();
            return;
        }
        this.mEndValues.f2916a.clear();
        this.mEndValues.f2917b.clear();
        this.mEndValues.f2918c.a();
    }

    public Animator createAnimator(ViewGroup viewGroup, p pVar, p pVar2) {
        return null;
    }

    public void createAnimators(ViewGroup viewGroup, q qVar, q qVar2, ArrayList<p> arrayList, ArrayList<p> arrayList2) {
        Animator createAnimator;
        int i;
        View view;
        Animator animator;
        p pVar;
        Animator animator2;
        p pVar2;
        b.f.a<Animator, d> runningAnimators = getRunningAnimators();
        SparseIntArray sparseIntArray = new SparseIntArray();
        int size = arrayList.size();
        int i2 = 0;
        while (i2 < size) {
            p pVar3 = arrayList.get(i2);
            p pVar4 = arrayList2.get(i2);
            if (pVar3 != null && !pVar3.f2915c.contains(this)) {
                pVar3 = null;
            }
            if (pVar4 != null && !pVar4.f2915c.contains(this)) {
                pVar4 = null;
            }
            if (pVar3 != null || pVar4 != null) {
                if ((pVar3 == null || pVar4 == null || isTransitionRequired(pVar3, pVar4)) && (createAnimator = createAnimator(viewGroup, pVar3, pVar4)) != null) {
                    if (pVar4 != null) {
                        View view2 = pVar4.f2914b;
                        String[] transitionProperties = getTransitionProperties();
                        if (transitionProperties != null && transitionProperties.length > 0) {
                            pVar2 = new p(view2);
                            p pVar5 = qVar2.f2916a.get(view2);
                            if (pVar5 != null) {
                                int i3 = 0;
                                while (i3 < transitionProperties.length) {
                                    pVar2.f2913a.put(transitionProperties[i3], pVar5.f2913a.get(transitionProperties[i3]));
                                    i3++;
                                    createAnimator = createAnimator;
                                    size = size;
                                    pVar5 = pVar5;
                                }
                            }
                            Animator animator3 = createAnimator;
                            i = size;
                            int i4 = runningAnimators.f1775h;
                            int i5 = 0;
                            while (true) {
                                if (i5 >= i4) {
                                    animator2 = animator3;
                                    break;
                                }
                                d dVar = runningAnimators.get(runningAnimators.h(i5));
                                if (dVar.f2893c != null && dVar.f2891a == view2 && dVar.f2892b.equals(getName()) && dVar.f2893c.equals(pVar2)) {
                                    animator2 = null;
                                    break;
                                }
                                i5++;
                            }
                        } else {
                            i = size;
                            animator2 = createAnimator;
                            pVar2 = null;
                        }
                        view = view2;
                        animator = animator2;
                        pVar = pVar2;
                    } else {
                        i = size;
                        view = pVar3.f2914b;
                        animator = createAnimator;
                        pVar = null;
                    }
                    if (animator == null) {
                        continue;
                    } else if (this.mPropagation == null) {
                        String name = getName();
                        y yVar = s.f2921a;
                        runningAnimators.put(animator, new d(view, name, this, new a0(viewGroup), pVar));
                        this.mAnimators.add(animator);
                    } else {
                        throw null;
                    }
                    i2++;
                    size = i;
                }
            }
            i = size;
            i2++;
            size = i;
        }
        if (sparseIntArray.size() != 0) {
            for (int i6 = 0; i6 < sparseIntArray.size(); i6++) {
                Animator animator4 = this.mAnimators.get(sparseIntArray.keyAt(i6));
                animator4.setStartDelay(animator4.getStartDelay() + (sparseIntArray.valueAt(i6) - RecyclerView.FOREVER_NS));
            }
        }
    }

    public void end() {
        int i = this.mNumInstances - 1;
        this.mNumInstances = i;
        if (i == 0) {
            ArrayList<f> arrayList = this.mListeners;
            if (arrayList != null && arrayList.size() > 0) {
                ArrayList arrayList2 = (ArrayList) this.mListeners.clone();
                int size = arrayList2.size();
                for (int i2 = 0; i2 < size; i2++) {
                    ((f) arrayList2.get(i2)).onTransitionEnd(this);
                }
            }
            for (int i3 = 0; i3 < this.mStartValues.f2918c.h(); i3++) {
                View i4 = this.mStartValues.f2918c.i(i3);
                if (i4 != null) {
                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                    i4.setHasTransientState(false);
                }
            }
            for (int i5 = 0; i5 < this.mEndValues.f2918c.h(); i5++) {
                View i6 = this.mEndValues.f2918c.i(i5);
                if (i6 != null) {
                    AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                    i6.setHasTransientState(false);
                }
            }
            this.mEnded = true;
        }
    }

    public j excludeChildren(View view, boolean z) {
        this.mTargetChildExcludes = excludeView(this.mTargetChildExcludes, view, z);
        return this;
    }

    public j excludeTarget(View view, boolean z) {
        this.mTargetExcludes = excludeView(this.mTargetExcludes, view, z);
        return this;
    }

    public void forceToEnd(ViewGroup viewGroup) {
        b.f.a<Animator, d> runningAnimators = getRunningAnimators();
        int i = runningAnimators.f1775h;
        if (viewGroup == null || i == 0) {
            return;
        }
        y yVar = s.f2921a;
        a0 a0Var = new a0(viewGroup);
        b.f.a aVar = new b.f.a(runningAnimators);
        runningAnimators.clear();
        for (int i2 = i - 1; i2 >= 0; i2--) {
            d dVar = (d) aVar.l(i2);
            if (dVar.f2891a != null && a0Var.equals(dVar.f2894d)) {
                ((Animator) aVar.h(i2)).end();
            }
        }
    }

    public long getDuration() {
        return this.mDuration;
    }

    public Rect getEpicenter() {
        e eVar = this.mEpicenterCallback;
        if (eVar == null) {
            return null;
        }
        return eVar.a(this);
    }

    public e getEpicenterCallback() {
        return this.mEpicenterCallback;
    }

    public TimeInterpolator getInterpolator() {
        return this.mInterpolator;
    }

    public p getMatchedTransitionValues(View view, boolean z) {
        n nVar = this.mParent;
        if (nVar != null) {
            return nVar.getMatchedTransitionValues(view, z);
        }
        ArrayList<p> arrayList = z ? this.mStartValuesList : this.mEndValuesList;
        if (arrayList == null) {
            return null;
        }
        int size = arrayList.size();
        int i = -1;
        int i2 = 0;
        while (true) {
            if (i2 >= size) {
                break;
            }
            p pVar = arrayList.get(i2);
            if (pVar == null) {
                return null;
            }
            if (pVar.f2914b == view) {
                i = i2;
                break;
            }
            i2++;
        }
        if (i >= 0) {
            return (z ? this.mEndValuesList : this.mStartValuesList).get(i);
        }
        return null;
    }

    public String getName() {
        return this.mName;
    }

    public b.z.e getPathMotion() {
        return this.mPathMotion;
    }

    public m getPropagation() {
        return this.mPropagation;
    }

    public long getStartDelay() {
        return this.mStartDelay;
    }

    public List<Integer> getTargetIds() {
        return this.mTargetIds;
    }

    public List<String> getTargetNames() {
        return this.mTargetNames;
    }

    public List<Class<?>> getTargetTypes() {
        return this.mTargetTypes;
    }

    public List<View> getTargets() {
        return this.mTargets;
    }

    public String[] getTransitionProperties() {
        return null;
    }

    public p getTransitionValues(View view, boolean z) {
        n nVar = this.mParent;
        if (nVar != null) {
            return nVar.getTransitionValues(view, z);
        }
        return (z ? this.mStartValues : this.mEndValues).f2916a.getOrDefault(view, null);
    }

    public boolean isTransitionRequired(p pVar, p pVar2) {
        if (pVar == null || pVar2 == null) {
            return false;
        }
        String[] transitionProperties = getTransitionProperties();
        if (transitionProperties != null) {
            for (String str : transitionProperties) {
                if (!isValueChanged(pVar, pVar2, str)) {
                }
            }
            return false;
        }
        for (String str2 : pVar.f2913a.keySet()) {
            if (isValueChanged(pVar, pVar2, str2)) {
            }
        }
        return false;
        return true;
    }

    public boolean isValidTarget(View view) {
        ArrayList<Class<?>> arrayList;
        ArrayList<String> arrayList2;
        int id = view.getId();
        ArrayList<Integer> arrayList3 = this.mTargetIdExcludes;
        if (arrayList3 == null || !arrayList3.contains(Integer.valueOf(id))) {
            ArrayList<View> arrayList4 = this.mTargetExcludes;
            if (arrayList4 == null || !arrayList4.contains(view)) {
                ArrayList<Class<?>> arrayList5 = this.mTargetTypeExcludes;
                if (arrayList5 != null) {
                    int size = arrayList5.size();
                    for (int i = 0; i < size; i++) {
                        if (this.mTargetTypeExcludes.get(i).isInstance(view)) {
                            return false;
                        }
                    }
                }
                if (this.mTargetNameExcludes != null) {
                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                    if (view.getTransitionName() != null && this.mTargetNameExcludes.contains(view.getTransitionName())) {
                        return false;
                    }
                }
                if ((this.mTargetIds.size() == 0 && this.mTargets.size() == 0 && (((arrayList = this.mTargetTypes) == null || arrayList.isEmpty()) && ((arrayList2 = this.mTargetNames) == null || arrayList2.isEmpty()))) || this.mTargetIds.contains(Integer.valueOf(id)) || this.mTargets.contains(view)) {
                    return true;
                }
                ArrayList<String> arrayList6 = this.mTargetNames;
                if (arrayList6 != null) {
                    AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                    if (arrayList6.contains(view.getTransitionName())) {
                        return true;
                    }
                }
                if (this.mTargetTypes != null) {
                    for (int i2 = 0; i2 < this.mTargetTypes.size(); i2++) {
                        if (this.mTargetTypes.get(i2).isInstance(view)) {
                            return true;
                        }
                    }
                }
                return false;
            }
            return false;
        }
        return false;
    }

    public void pause(View view) {
        if (this.mEnded) {
            return;
        }
        b.f.a<Animator, d> runningAnimators = getRunningAnimators();
        int i = runningAnimators.f1775h;
        y yVar = s.f2921a;
        a0 a0Var = new a0(view);
        for (int i2 = i - 1; i2 >= 0; i2--) {
            d l = runningAnimators.l(i2);
            if (l.f2891a != null && a0Var.equals(l.f2894d)) {
                runningAnimators.h(i2).pause();
            }
        }
        ArrayList<f> arrayList = this.mListeners;
        if (arrayList != null && arrayList.size() > 0) {
            ArrayList arrayList2 = (ArrayList) this.mListeners.clone();
            int size = arrayList2.size();
            for (int i3 = 0; i3 < size; i3++) {
                ((f) arrayList2.get(i3)).onTransitionPause(this);
            }
        }
        this.mPaused = true;
    }

    public void playTransition(ViewGroup viewGroup) {
        d orDefault;
        this.mStartValuesList = new ArrayList<>();
        this.mEndValuesList = new ArrayList<>();
        matchStartAndEnd(this.mStartValues, this.mEndValues);
        b.f.a<Animator, d> runningAnimators = getRunningAnimators();
        int i = runningAnimators.f1775h;
        y yVar = s.f2921a;
        a0 a0Var = new a0(viewGroup);
        for (int i2 = i - 1; i2 >= 0; i2--) {
            Animator h2 = runningAnimators.h(i2);
            if (h2 != null && (orDefault = runningAnimators.getOrDefault(h2, null)) != null && orDefault.f2891a != null && a0Var.equals(orDefault.f2894d)) {
                p pVar = orDefault.f2893c;
                View view = orDefault.f2891a;
                p transitionValues = getTransitionValues(view, true);
                p matchedTransitionValues = getMatchedTransitionValues(view, true);
                if (transitionValues == null && matchedTransitionValues == null) {
                    matchedTransitionValues = this.mEndValues.f2916a.get(view);
                }
                if (!(transitionValues == null && matchedTransitionValues == null) && orDefault.f2895e.isTransitionRequired(pVar, matchedTransitionValues)) {
                    if (!h2.isRunning() && !h2.isStarted()) {
                        runningAnimators.remove(h2);
                    } else {
                        h2.cancel();
                    }
                }
            }
        }
        createAnimators(viewGroup, this.mStartValues, this.mEndValues, this.mStartValuesList, this.mEndValuesList);
        runAnimators();
    }

    public j removeListener(f fVar) {
        ArrayList<f> arrayList = this.mListeners;
        if (arrayList == null) {
            return this;
        }
        arrayList.remove(fVar);
        if (this.mListeners.size() == 0) {
            this.mListeners = null;
        }
        return this;
    }

    public j removeTarget(View view) {
        this.mTargets.remove(view);
        return this;
    }

    public void resume(View view) {
        if (this.mPaused) {
            if (!this.mEnded) {
                b.f.a<Animator, d> runningAnimators = getRunningAnimators();
                int i = runningAnimators.f1775h;
                y yVar = s.f2921a;
                a0 a0Var = new a0(view);
                for (int i2 = i - 1; i2 >= 0; i2--) {
                    d l = runningAnimators.l(i2);
                    if (l.f2891a != null && a0Var.equals(l.f2894d)) {
                        runningAnimators.h(i2).resume();
                    }
                }
                ArrayList<f> arrayList = this.mListeners;
                if (arrayList != null && arrayList.size() > 0) {
                    ArrayList arrayList2 = (ArrayList) this.mListeners.clone();
                    int size = arrayList2.size();
                    for (int i3 = 0; i3 < size; i3++) {
                        ((f) arrayList2.get(i3)).onTransitionResume(this);
                    }
                }
            }
            this.mPaused = false;
        }
    }

    public void runAnimators() {
        start();
        b.f.a<Animator, d> runningAnimators = getRunningAnimators();
        Iterator<Animator> it = this.mAnimators.iterator();
        while (it.hasNext()) {
            Animator next = it.next();
            if (runningAnimators.containsKey(next)) {
                start();
                runAnimator(next, runningAnimators);
            }
        }
        this.mAnimators.clear();
        end();
    }

    public void setCanRemoveViews(boolean z) {
        this.mCanRemoveViews = z;
    }

    public j setDuration(long j) {
        this.mDuration = j;
        return this;
    }

    public void setEpicenterCallback(e eVar) {
        this.mEpicenterCallback = eVar;
    }

    public j setInterpolator(TimeInterpolator timeInterpolator) {
        this.mInterpolator = timeInterpolator;
        return this;
    }

    public void setMatchOrder(int... iArr) {
        if (iArr != null && iArr.length != 0) {
            for (int i = 0; i < iArr.length; i++) {
                if (isValidMatch(iArr[i])) {
                    if (alreadyContains(iArr, i)) {
                        throw new IllegalArgumentException("matches contains a duplicate value");
                    }
                } else {
                    throw new IllegalArgumentException("matches contains invalid value");
                }
            }
            this.mMatchOrder = (int[]) iArr.clone();
            return;
        }
        this.mMatchOrder = DEFAULT_MATCH_ORDER;
    }

    public void setPathMotion(b.z.e eVar) {
        if (eVar == null) {
            this.mPathMotion = STRAIGHT_PATH_MOTION;
        } else {
            this.mPathMotion = eVar;
        }
    }

    public void setPropagation(m mVar) {
    }

    public j setSceneRoot(ViewGroup viewGroup) {
        this.mSceneRoot = viewGroup;
        return this;
    }

    public j setStartDelay(long j) {
        this.mStartDelay = j;
        return this;
    }

    public void start() {
        if (this.mNumInstances == 0) {
            ArrayList<f> arrayList = this.mListeners;
            if (arrayList != null && arrayList.size() > 0) {
                ArrayList arrayList2 = (ArrayList) this.mListeners.clone();
                int size = arrayList2.size();
                for (int i = 0; i < size; i++) {
                    ((f) arrayList2.get(i)).onTransitionStart(this);
                }
            }
            this.mEnded = false;
        }
        this.mNumInstances++;
    }

    public String toString() {
        return toString("");
    }

    public j addTarget(int i) {
        if (i != 0) {
            this.mTargetIds.add(Integer.valueOf(i));
        }
        return this;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // 
    /* renamed from: clone */
    public j mo0clone() {
        try {
            j jVar = (j) super.clone();
            jVar.mAnimators = new ArrayList<>();
            jVar.mStartValues = new q();
            jVar.mEndValues = new q();
            jVar.mStartValuesList = null;
            jVar.mEndValuesList = null;
            return jVar;
        } catch (CloneNotSupportedException unused) {
            return null;
        }
    }

    public j excludeChildren(int i, boolean z) {
        this.mTargetIdChildExcludes = excludeId(this.mTargetIdChildExcludes, i, z);
        return this;
    }

    public j excludeTarget(int i, boolean z) {
        this.mTargetIdExcludes = excludeId(this.mTargetIdExcludes, i, z);
        return this;
    }

    public j removeTarget(int i) {
        if (i != 0) {
            this.mTargetIds.remove(Integer.valueOf(i));
        }
        return this;
    }

    public String toString(String str) {
        StringBuilder x = c.b.a.a.a.x(str);
        x.append(getClass().getSimpleName());
        x.append("@");
        x.append(Integer.toHexString(hashCode()));
        x.append(": ");
        String sb = x.toString();
        if (this.mDuration != -1) {
            StringBuilder A = c.b.a.a.a.A(sb, "dur(");
            A.append(this.mDuration);
            A.append(") ");
            sb = A.toString();
        }
        if (this.mStartDelay != -1) {
            StringBuilder A2 = c.b.a.a.a.A(sb, "dly(");
            A2.append(this.mStartDelay);
            A2.append(") ");
            sb = A2.toString();
        }
        if (this.mInterpolator != null) {
            StringBuilder A3 = c.b.a.a.a.A(sb, "interp(");
            A3.append(this.mInterpolator);
            A3.append(") ");
            sb = A3.toString();
        }
        if (this.mTargetIds.size() > 0 || this.mTargets.size() > 0) {
            String q = c.b.a.a.a.q(sb, "tgts(");
            if (this.mTargetIds.size() > 0) {
                for (int i = 0; i < this.mTargetIds.size(); i++) {
                    if (i > 0) {
                        q = c.b.a.a.a.q(q, ", ");
                    }
                    StringBuilder x2 = c.b.a.a.a.x(q);
                    x2.append(this.mTargetIds.get(i));
                    q = x2.toString();
                }
            }
            if (this.mTargets.size() > 0) {
                for (int i2 = 0; i2 < this.mTargets.size(); i2++) {
                    if (i2 > 0) {
                        q = c.b.a.a.a.q(q, ", ");
                    }
                    StringBuilder x3 = c.b.a.a.a.x(q);
                    x3.append(this.mTargets.get(i2));
                    q = x3.toString();
                }
            }
            return c.b.a.a.a.q(q, ")");
        }
        return sb;
    }

    public j addTarget(String str) {
        if (this.mTargetNames == null) {
            this.mTargetNames = new ArrayList<>();
        }
        this.mTargetNames.add(str);
        return this;
    }

    public j excludeChildren(Class<?> cls, boolean z) {
        this.mTargetTypeChildExcludes = excludeType(this.mTargetTypeChildExcludes, cls, z);
        return this;
    }

    public j excludeTarget(String str, boolean z) {
        this.mTargetNameExcludes = excludeObject(this.mTargetNameExcludes, str, z);
        return this;
    }

    public j removeTarget(String str) {
        ArrayList<String> arrayList = this.mTargetNames;
        if (arrayList != null) {
            arrayList.remove(str);
        }
        return this;
    }

    public j excludeTarget(Class<?> cls, boolean z) {
        this.mTargetTypeExcludes = excludeType(this.mTargetTypeExcludes, cls, z);
        return this;
    }

    public j removeTarget(Class<?> cls) {
        ArrayList<Class<?>> arrayList = this.mTargetTypes;
        if (arrayList != null) {
            arrayList.remove(cls);
        }
        return this;
    }

    public j addTarget(Class<?> cls) {
        if (this.mTargetTypes == null) {
            this.mTargetTypes = new ArrayList<>();
        }
        this.mTargetTypes.add(cls);
        return this;
    }

    @SuppressLint({"RestrictedApi"})
    public j(Context context, AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, i.f2886a);
        XmlResourceParser xmlResourceParser = (XmlResourceParser) attributeSet;
        long b2 = b.j.c.b.f.b(obtainStyledAttributes, xmlResourceParser, "duration", 1, -1);
        if (b2 >= 0) {
            setDuration(b2);
        }
        long b3 = b.j.c.b.f.b(obtainStyledAttributes, xmlResourceParser, "startDelay", 2, -1);
        if (b3 > 0) {
            setStartDelay(b3);
        }
        int c2 = b.j.c.b.f.c(obtainStyledAttributes, xmlResourceParser, "interpolator", 0, 0);
        if (c2 > 0) {
            setInterpolator(AnimationUtils.loadInterpolator(context, c2));
        }
        String d2 = b.j.c.b.f.d(obtainStyledAttributes, xmlResourceParser, "matchOrder", 3);
        if (d2 != null) {
            setMatchOrder(parseMatchOrder(d2));
        }
        obtainStyledAttributes.recycle();
    }
}