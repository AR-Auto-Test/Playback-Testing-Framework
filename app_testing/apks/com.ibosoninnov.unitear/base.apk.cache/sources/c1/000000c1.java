package androidx.media;

import b.b0.a;
import java.util.Objects;

/* loaded from: classes.dex */
public final class AudioAttributesCompatParcelizer {
    public static AudioAttributesCompat read(a aVar) {
        AudioAttributesCompat audioAttributesCompat = new AudioAttributesCompat();
        Object obj = audioAttributesCompat.f335b;
        if (aVar.i(1)) {
            obj = aVar.o();
        }
        audioAttributesCompat.f335b = (AudioAttributesImpl) obj;
        return audioAttributesCompat;
    }

    public static void write(AudioAttributesCompat audioAttributesCompat, a aVar) {
        Objects.requireNonNull(aVar);
        AudioAttributesImpl audioAttributesImpl = audioAttributesCompat.f335b;
        aVar.p(1);
        aVar.w(audioAttributesImpl);
    }
}