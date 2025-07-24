package b.z;

import android.animation.TimeInterpolator;
import android.util.AndroidRuntimeException;
import android.view.View;
import android.view.ViewGroup;
import b.z.j;
import java.util.ArrayList;
import java.util.Iterator;

/* compiled from: TransitionSet.java */
/* loaded from: classes.dex */
public class n extends j {

    /* renamed from: d  reason: collision with root package name */
    public int f2905d;

    /* renamed from: b  reason: collision with root package name */
    public ArrayList<j> f2903b = new ArrayList<>();

    /* renamed from: c  reason: collision with root package name */
    public boolean f2904c = true;

    /* renamed from: e  reason: collision with root package name */
    public boolean f2906e = false;

    /* renamed from: f  reason: collision with root package name */
    public int f2907f = 0;

    /* compiled from: TransitionSet.java */
    /* loaded from: classes.dex */
    public class a extends k {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ j f2908a;

        public a(n nVar, j jVar) {
            this.f2908a = jVar;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            this.f2908a.runAnimators();
            jVar.removeListener(this);
        }
    }

    /* compiled from: TransitionSet.java */
    /* loaded from: classes.dex */
    public static class b extends k {

        /* renamed from: a  reason: collision with root package name */
        public n f2909a;

        public b(n nVar) {
            this.f2909a = nVar;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            n nVar = this.f2909a;
            int i = nVar.f2905d - 1;
            nVar.f2905d = i;
            if (i == 0) {
                nVar.f2906e = false;
                nVar.end();
            }
            jVar.removeListener(this);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionStart(j jVar) {
            n nVar = this.f2909a;
            if (nVar.f2906e) {
                return;
            }
            nVar.start();
            this.f2909a.f2906e = true;
        }
    }

    public n a(j jVar) {
        this.f2903b.add(jVar);
        jVar.mParent = this;
        long j = this.mDuration;
        if (j >= 0) {
            jVar.setDuration(j);
        }
        if ((this.f2907f & 1) != 0) {
            jVar.setInterpolator(getInterpolator());
        }
        if ((this.f2907f & 2) != 0) {
            jVar.setPropagation(getPropagation());
        }
        if ((this.f2907f & 4) != 0) {
            jVar.setPathMotion(getPathMotion());
        }
        if ((this.f2907f & 8) != 0) {
            jVar.setEpicenterCallback(getEpicenterCallback());
        }
        return this;
    }

    @Override // b.z.j
    public j addListener(j.f fVar) {
        return (n) super.addListener(fVar);
    }

    @Override // b.z.j
    public j addTarget(View view) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).addTarget(view);
        }
        return (n) super.addTarget(view);
    }

    public j b(int i) {
        if (i < 0 || i >= this.f2903b.size()) {
            return null;
        }
        return this.f2903b.get(i);
    }

    public n c(long j) {
        ArrayList<j> arrayList;
        super.setDuration(j);
        if (this.mDuration >= 0 && (arrayList = this.f2903b) != null) {
            int size = arrayList.size();
            for (int i = 0; i < size; i++) {
                this.f2903b.get(i).setDuration(j);
            }
        }
        return this;
    }

    @Override // b.z.j
    public void cancel() {
        super.cancel();
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).cancel();
        }
    }

    @Override // b.z.j
    public void captureEndValues(p pVar) {
        if (isValidTarget(pVar.f2914b)) {
            Iterator<j> it = this.f2903b.iterator();
            while (it.hasNext()) {
                j next = it.next();
                if (next.isValidTarget(pVar.f2914b)) {
                    next.captureEndValues(pVar);
                    pVar.f2915c.add(next);
                }
            }
        }
    }

    @Override // b.z.j
    public void capturePropagationValues(p pVar) {
        super.capturePropagationValues(pVar);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).capturePropagationValues(pVar);
        }
    }

    @Override // b.z.j
    public void captureStartValues(p pVar) {
        if (isValidTarget(pVar.f2914b)) {
            Iterator<j> it = this.f2903b.iterator();
            while (it.hasNext()) {
                j next = it.next();
                if (next.isValidTarget(pVar.f2914b)) {
                    next.captureStartValues(pVar);
                    pVar.f2915c.add(next);
                }
            }
        }
    }

    @Override // b.z.j
    public void createAnimators(ViewGroup viewGroup, q qVar, q qVar2, ArrayList<p> arrayList, ArrayList<p> arrayList2) {
        long startDelay = getStartDelay();
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            j jVar = this.f2903b.get(i);
            if (startDelay > 0 && (this.f2904c || i == 0)) {
                long startDelay2 = jVar.getStartDelay();
                if (startDelay2 > 0) {
                    jVar.setStartDelay(startDelay2 + startDelay);
                } else {
                    jVar.setStartDelay(startDelay);
                }
            }
            jVar.createAnimators(viewGroup, qVar, qVar2, arrayList, arrayList2);
        }
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // b.z.j
    /* renamed from: d */
    public n setInterpolator(TimeInterpolator timeInterpolator) {
        this.f2907f |= 1;
        ArrayList<j> arrayList = this.f2903b;
        if (arrayList != null) {
            int size = arrayList.size();
            for (int i = 0; i < size; i++) {
                this.f2903b.get(i).setInterpolator(timeInterpolator);
            }
        }
        return (n) super.setInterpolator(timeInterpolator);
    }

    public n e(int i) {
        if (i == 0) {
            this.f2904c = true;
        } else if (i == 1) {
            this.f2904c = false;
        } else {
            throw new AndroidRuntimeException(c.b.a.a.a.j("Invalid parameter for TransitionSet ordering: ", i));
        }
        return this;
    }

    @Override // b.z.j
    public j excludeTarget(View view, boolean z) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).excludeTarget(view, z);
        }
        return super.excludeTarget(view, z);
    }

    @Override // b.z.j
    public void forceToEnd(ViewGroup viewGroup) {
        super.forceToEnd(viewGroup);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).forceToEnd(viewGroup);
        }
    }

    @Override // b.z.j
    public void pause(View view) {
        super.pause(view);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).pause(view);
        }
    }

    @Override // b.z.j
    public j removeListener(j.f fVar) {
        return (n) super.removeListener(fVar);
    }

    @Override // b.z.j
    public j removeTarget(int i) {
        for (int i2 = 0; i2 < this.f2903b.size(); i2++) {
            this.f2903b.get(i2).removeTarget(i);
        }
        return (n) super.removeTarget(i);
    }

    @Override // b.z.j
    public void resume(View view) {
        super.resume(view);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).resume(view);
        }
    }

    @Override // b.z.j
    public void runAnimators() {
        if (this.f2903b.isEmpty()) {
            start();
            end();
            return;
        }
        b bVar = new b(this);
        Iterator<j> it = this.f2903b.iterator();
        while (it.hasNext()) {
            it.next().addListener(bVar);
        }
        this.f2905d = this.f2903b.size();
        if (!this.f2904c) {
            for (int i = 1; i < this.f2903b.size(); i++) {
                this.f2903b.get(i - 1).addListener(new a(this, this.f2903b.get(i)));
            }
            j jVar = this.f2903b.get(0);
            if (jVar != null) {
                jVar.runAnimators();
                return;
            }
            return;
        }
        Iterator<j> it2 = this.f2903b.iterator();
        while (it2.hasNext()) {
            it2.next().runAnimators();
        }
    }

    @Override // b.z.j
    public void setCanRemoveViews(boolean z) {
        super.setCanRemoveViews(z);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).setCanRemoveViews(z);
        }
    }

    @Override // b.z.j
    public /* bridge */ /* synthetic */ j setDuration(long j) {
        c(j);
        return this;
    }

    @Override // b.z.j
    public void setEpicenterCallback(j.e eVar) {
        super.setEpicenterCallback(eVar);
        this.f2907f |= 8;
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).setEpicenterCallback(eVar);
        }
    }

    @Override // b.z.j
    public void setPathMotion(e eVar) {
        super.setPathMotion(eVar);
        this.f2907f |= 4;
        if (this.f2903b != null) {
            for (int i = 0; i < this.f2903b.size(); i++) {
                this.f2903b.get(i).setPathMotion(eVar);
            }
        }
    }

    @Override // b.z.j
    public void setPropagation(m mVar) {
        super.setPropagation(mVar);
        this.f2907f |= 2;
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).setPropagation(mVar);
        }
    }

    @Override // b.z.j
    public j setSceneRoot(ViewGroup viewGroup) {
        super.setSceneRoot(viewGroup);
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            this.f2903b.get(i).setSceneRoot(viewGroup);
        }
        return this;
    }

    @Override // b.z.j
    public j setStartDelay(long j) {
        return (n) super.setStartDelay(j);
    }

    @Override // b.z.j
    public String toString(String str) {
        String jVar = super.toString(str);
        for (int i = 0; i < this.f2903b.size(); i++) {
            StringBuilder A = c.b.a.a.a.A(jVar, "\n");
            A.append(this.f2903b.get(i).toString(str + "  "));
            jVar = A.toString();
        }
        return jVar;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // b.z.j
    /* renamed from: clone */
    public j mo0clone() {
        n nVar = (n) super.mo0clone();
        nVar.f2903b = new ArrayList<>();
        int size = this.f2903b.size();
        for (int i = 0; i < size; i++) {
            j mo0clone = this.f2903b.get(i).mo0clone();
            nVar.f2903b.add(mo0clone);
            mo0clone.mParent = nVar;
        }
        return nVar;
    }

    @Override // b.z.j
    public j addTarget(int i) {
        for (int i2 = 0; i2 < this.f2903b.size(); i2++) {
            this.f2903b.get(i2).addTarget(i);
        }
        return (n) super.addTarget(i);
    }

    @Override // b.z.j
    public j excludeTarget(String str, boolean z) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).excludeTarget(str, z);
        }
        return super.excludeTarget(str, z);
    }

    @Override // b.z.j
    public j removeTarget(View view) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).removeTarget(view);
        }
        return (n) super.removeTarget(view);
    }

    @Override // b.z.j
    public j addTarget(String str) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).addTarget(str);
        }
        return (n) super.addTarget(str);
    }

    @Override // b.z.j
    public j excludeTarget(int i, boolean z) {
        for (int i2 = 0; i2 < this.f2903b.size(); i2++) {
            this.f2903b.get(i2).excludeTarget(i, z);
        }
        return super.excludeTarget(i, z);
    }

    @Override // b.z.j
    public j removeTarget(Class cls) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).removeTarget(cls);
        }
        return (n) super.removeTarget(cls);
    }

    @Override // b.z.j
    public j addTarget(Class cls) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).addTarget(cls);
        }
        return (n) super.addTarget(cls);
    }

    @Override // b.z.j
    public j excludeTarget(Class<?> cls, boolean z) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).excludeTarget(cls, z);
        }
        return super.excludeTarget(cls, z);
    }

    @Override // b.z.j
    public j removeTarget(String str) {
        for (int i = 0; i < this.f2903b.size(); i++) {
            this.f2903b.get(i).removeTarget(str);
        }
        return (n) super.removeTarget(str);
    }
}