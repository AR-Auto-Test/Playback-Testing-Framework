"""
AR Evaluation Prompts Library

This module provides standardized prompts and descriptions for AR application evaluation.
It centralizes all text-based components used across different model evaluation scripts.
"""

def generate_system_prompt():
    """Generate the system prompt for AR evaluation."""
    return """You are a specialized AI assistant trained to review AR applications based on screen recordings. The video provided is a screen recording from a mobile phone that demonstrates an AR (Augmented Reality) application. We specifically designed and selected certain environments and use cases to test the AR application to identify any potential performance issues with AR functionality.

    In all the screen recording videos, the movement of AR objects is simulated programmatically as interactions on the phone screen. Currently, the simulated interactions include the following two types:
    1. Tapping the screen: The AR object will immediately appear at the corresponding position.
    2. Swiping the screen: The AR object will move along the swipe path to the corresponding positions.
    
    Your role:
    • Assess the AR application’s performance on six key metrics:  
        1. Object Placement: Evaluate how accurately AR objects are positioned relative to ALL environmental elements (floors, walls, tables, etc.). Focus ONLY on physical positioning, not visual integration through shadows. Check for floating (object not touching surface), sinking (object partially embedded in surface), or intersection with other objects (wall, furniture). Pay special attention to edges and boundaries where placement issues are most visible, and whether objects extend beyond supporting surfaces in a physically implausible way (center of gravity issues).
        2. Object Movement: Distinguish between camera movement and object movement. Only evaluate object movement when the AR object itself moves relative to its environment (through swipes or animations), not when perspective changes due to camera movement. Assess both responsiveness to user interaction and stability during movement.
        3. Occlusion: Evaluate whether AR objects correctly appear in front of or behind real objects based on their spatial positioning. Before evaluating performance, first determine if the video contains scenarios that test occlusion (objects passing between camera and AR object). If no occlusion testing is present, clearly state this rather than assuming it works correctly.
        4. Lighting: Evaluate lighting within the specific environmental context. First assess the ambient lighting conditions (bright, dim, indoor, outdoor), then evaluate whether shadows, highlights, and reflections are appropriate for that environment. In dimly lit or shaded environments, minimal shadows may be acceptable.
        5. Visual Artifacts and Rendering Issues: Systematically check for specific rendering issues including jagged edges, texture problems, polygon issues, flickering, Z-fighting, and transparency errors. Distinguish between rendering issues and lighting issues.
        6. Black Screen: Check for complete or partial black screens, screen freezing, or significant rendering failures. Distinguish between intentional UI elements (like loading screens) and technical issues.

    • You could ignore the following aspects in evaluation:
        1. The size of AR object: Ignore the size of AR objects as resizing features may be user-adjustable.
        2. Stylized or Cartoonish AR Objects: Some apps may provide stylistic and cartoonish models according to the theme of the app. Stylization does not account for realism issues.
        3. The UI elements: Disregard UI elements (grid lines, measures, etc.) as their design and interaction with AR objects are developer-defined UI/UX choices and outside the scope of AR functionality evaluation.
        
    • If you do see any issues in any extracted frames from a video, thoroughly explain which frames or angles (if known) suggest that problem (e.g., object floating above the floor, or missing partial occlusion behind a table leg). Conversely, if the video does not provide enough evidence, clarify that you have “No Issue” but the scenario was not fully tested. 

    • For each evaluation question, provide your response in JSON format. Ensure it is a valid JSON object adhering to this format:
    {
        "Video_name": string,   
        "Metrics": string, // the metric to evaluate, e.g. Object Placement
        "Issue": boolean, // true if an issue is found, false if not or cannot decide 
        "Reason":  string //The reason why you think there are issues or no issues. Provide explanation.
    }
    """

def generate_ar_metrics_description(metric_index):
    """Generate detailed description for a specific AR metric by index.
    
    Args:
        metric_index: Integer from 0-5 representing the metric index
            0 = Object Placement
            1 = Object Movement
            2 = Occlusion
            3 = Lighting
            4 = Visual Artifacts and Rendering Issues
            5 = Black Screen
    
    Returns:
        String containing the detailed description for the specified metric
    """
    metrics_descriptions = [
        # 0. Object Placement
        """
        Evaluate whether AR objects are positioned correctly in the real environment by checking ALL of the following criteria:
        a. Contact accuracy – AR object should *visibly* touch its support plane; a gap lower than 3 pixels at 1080p (approximately 1–2 cm in real scale) is acceptable.  
        b. Intersection – any penetration through floor, wall, or furniture greater than 3 pixels for greater or equal to 3 consecutive frames is a defect.  
        c. Physical support – greater or equal to 50% of the object’s footprint (or centre-of-mass) must lie over the supporting surface; bigger overhang → defect.  
        d. Check all axes: up or down, left or right, forward or back (e.g., object clipping into rear wall).  
        e. Ignore temporary UI placement rings or cursors.
        """,
        
        # 1. Object Movement
        """
        When evaluating movement, carefully distinguish between:
        1. OBJECT MOVEMENT: When the AR object itself moves relative to the environment (through user interaction or animation)
        2. CAMERA MOVEMENT: When perspective changes but the object remains fixed in the environment

        Evaluate ONLY object movement (not camera movement) using these criteria:
        - Movement appearing fluid without stuttering, jumping, or discontinuity
        - Objects responding appropriately to user interactions (taps, swipes)
        - Objects maintaining correct spatial relationships with the scene while moving
        - Objects stabilizing after movement stops without drifting or oscillating

        If the object is STATIC throughout the video (no movement relative to environment):
        - Note this fact and evaluate whether this appears intentional
        - Check if the object maintains stable positioning despite camera movement
        """,
        
        # 2. Occlusion
        """
        FIRST DETERMINE: Does the video contain any scenarios that actually test occlusion? 
        Occlusion testing happens when:
        - Real objects pass between the camera and AR object (should obscure the AR object)
        - AR objects pass behind real objects from the camera's perspective
        - People or moving objects interact with the AR object

        Only if occlusion is being tested, evaluate:
        - Virtual objects being properly hidden when they should be occluded by real objects
            -- Partial occlusion: Virtual objects must correctly appear partially hidden, with precise occlusion edges and no visible clipping errors or unnatural gaps.
            -- Complete occlusion: Virtual objects must be completely hidden when fully occluded by real-world objects without visual artifacts or errors.
        - Occlusion edges appearing precise without obvious clipping errors or gaps
        - People or moving objects in the environment correctly occluding AR objects
        - Objects at different distances showing correct front/back occlusion relationships
        - Occlusion effects remaining stable without suddenly appearing or disappearing with perspective changes
        - Occlusion relationships updating dynamically when AR objects move and interact with the environment

        If no occlusion testing scenarios occur in the video:
        - Clearly state that occlusion cannot be properly evaluated due to lack of testing scenarios
        """,
        
        # 3. Lighting
        """
        FIRST ASSESS THE ENVIRONMENT:
        Identify the lighting environment before evaluating AR objects. Environments are categorized by overall brightness and how shadows appear. Use simple categories to describe the scene’s lighting:
        - Bright Environment: Very well-lit (e.g., direct sunlight or strong indoor lighting). Surfaces are clearly illuminated. Distinct shadows are present and well-defined, indicating a strong light source direction. Colors appear vibrant, and highlights on objects are intense.
        - Moderate Environment: Evenly lit with adequate light (e.g., daytime indoors or cloudy outdoor). Lighting is neither harsh nor dim. Shadows exist but are soft or moderately defined. Objects and surfaces are clearly visible with natural color tones.
        - Shaded Environment: Light is mostly indirect or diffused (e.g., in shade outdoors or under diffuse indoor light). Brightness is reduced and shadows are minimal or very soft, often just gentle darkening under objects. The overall scene has muted highlights and a softer contrast.
        - Dim Environment: Very low light (e.g., dusk, nighttime indoors with few lights). Objects are barely illuminated; colors may appear muted or gray. Shadows may be absent or extremely faint because there is little to no direct light. The scene might have a grainy or low-contrast appearance due to minimal lighting.
        Ensure the environment type is clearly identified as one of the above categories. Avoid ambiguous labels. This classification will determine the expected lighting behavior for AR objects in the next step.

        THEN EVALUATE AR OBJECT LIGHTING:
        For the identified environment, evaluate the AR object’s lighting to see if it blends realistically. Each aspect of the virtual object’s lighting (brightness, shadows, highlights, reflections) should match the environment category:
        - Brightness Matching: The AR object’s overall brightness should match the scene’s illumination level. In a Bright environment, the object should appear well-lit with vibrant colors. In a Moderate setting, it should have balanced light – not too bright or too dark. In a Shaded scene, the object should look slightly subdued (no overly bright spots). In a Dim environment, the object may appear darker or muted, aligning with the low ambient light.
        - Shadow Presence & Intensity: Check if the AR object casts a shadow and if so, whether it matches the environment’s lighting:
            -- In Bright environments, a clear, well-defined shadow is required. The shadow should fall in the correct direction relative to the real light source and have an intensity similar to real object shadows in the scene. A missing shadow in a bright scene is unacceptable, as it will make the object look disconnected.
            -- In Moderate environments, the AR object should cast a soft shadow. The shadow should be visible but slightly blurry or lighter-edged, consistent with moderately bright ambient light. Its presence is important, though it can be less intense than in bright sun.
            -- In Shaded environments, minimal or very soft shadows are expected. It is acceptable if the AR object’s shadow is faint or diffuse. A strong shadow would be problematic here because real shadows are weak in shade. Ensure any subtle shadow aligns with the diffused light direction (if discernible).
            -- In Dim environments, shadows may be absent or extremely faint. This is acceptable due to the low light. Do not penalize an AR object for lacking a shadow in very dark scenes, as real objects under such conditions also cast hardly any visible shadow. However, if a shadow is visible, it should be very soft and light.
        - Highlight & Reflection Consistency: Evaluate the highlights (specular spots) and reflections on the AR object’s surface:
            -- They should correspond to the environment’s light sources. In Bright or Moderate scenes, the object might have visible highlights where light hits (e.g., a shiny spot on a reflective object). These highlights must appear on the correct side of the object (same side as the light) and not be too bright in moderate settings.
            -- In Shaded environments, only gentle or no obvious highlights should appear, since light is indirect. The object should not look glossy as if under direct light when it’s in a shadowed area.
            -- In Dim environments, little to no specular highlight should be present, because the scene has almost no direct light sources. If the object is reflective, it might show very muted reflections (or none visible) to match the dark setting.
            -- Reflections on the AR object (if it has reflective material) should mirror the scene’s brightness and color tone. For example, in a bright blue sky environment, a metallic AR object might pick up a bluish tint in its reflections. In a dim, yellow-lit room, the object’s reflections should be dark and tinged warm, not bright white.
        - Color and Tone Adaptation: The AR object’s color tone should adapt to the environment lighting. In bright, cool lighting (e.g., daylight), the object may appear slightly cooler in tone; in warm indoor lighting, the object might take on a warmer hue. In low light, colors may desaturate. The goal is that the object looks like it belongs in the scene, without overly vibrant colors in a dim setting or dull colors in a bright setting.
        - Consistency with Real Objects: Compare the AR object with real objects in the camera view:
            -- Shadows alignment: If real objects in the scene have shadows, the AR object’s shadow should fall in the same direction and similar softness. If real objects have no visible shadows (e.g., in dim light), the AR object should similarly lack a strong shadow.
            -- Lighting direction on object: The side of the AR object that is lit (bright side) and the side that is in shade should correspond to the environment’s light direction. For instance, if the room’s lamp lights real objects from the right, the AR object should be brighter on the right side and shade on the left.
            -- Shadow contact: In brighter scenarios, ensure the AR object’s shadow connects with the object at the base and looks attached (no floating effect). The shadow should also sit correctly on the real surface (ground or table) without misalignment.
        """,
        
        # 4. Visual Artifacts and Rendering Issues
        """
        Systematically check for these specific rendering issues:
        1. EDGE QUALITY: Are object edges free from jaggedness, blurring, or flickering?
        2. TEXTURES: Are textures loading correctly without display errors or distortion?
        3. COLORS: Are objects free from unnatural shine, oversaturation, or color bleeding?
        4. STABILITY: Are objects free from flickering, jittering, or other unstable visual effects?
        5. MATERIALS: Are object materials displaying correctly (e.g., metals having metallic sheen, glass having transparency)?
        6. GEOMETRY: Are objects free from unnatural geometric deformation or stretching?
        7. ARTIFACTS: Are there any visual glitches, polygon issues, Z-fighting, or image corruption?

        Note: Distinguish between rendering issues (which impact the visual quality of the model itself) and lighting issues (which relate to environmental integration through shadows and highlights).
        """,
        
        # 5. Black Screen
        """
        Systematically check for these display problems:
        1. BLACK SCREENS: Is the screen completely or partially black at any point?
        2. DISAPPEARANCE: Do AR objects suddenly disappear unexpectedly?
        3. ABNORMAL COLORS: Does the screen display large areas of white or other abnormal colors?
        4. FREEZING: Does the screen freeze or fail to update in large areas?
        5. CRASHES: Does the AR application crash or display abnormally after certain interactions?
        6. COLOR ARTIFACTS: Does the screen display unintended color blocks or patterns?
        7. CAMERA ISSUES: Does the camera feed fail to display properly or exhibit severe delay?
        8. SYNCHRONIZATION: Are there obvious synchronization issues between AR layer and camera layer?
        """
    ]
    
    if metric_index < 0 or metric_index >= len(metrics_descriptions):
        raise ValueError(f"Invalid metric index: {metric_index}. Must be between 0 and {len(metrics_descriptions)-1}")
    
    return metrics_descriptions[metric_index]

def generate_conversation_questions(include_descriptions=False):
    """Generate a list of conversation questions for AR evaluation.
    
    Args:
        include_descriptions: If True, includes detailed metric descriptions in questions
        
    Returns:
        List of strings containing the conversation questions
    """
    metrics = [
        "Object Placement",
        "Object Movement", 
        "Occlusion",
        "Lighting",
        "Visual Artifacts and Rendering Issues", 
        "Black Screen"
    ]
    
    # Initial verification questions
    conversation = [
        "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video? can you describe the setting (indoor/outdoor), the primary surfaces or objects in view, and any notable lighting conditions? You don't need to do the evaluation here now.",
        "What is the 3D object in the video? If you can see the 3D object, describe it in detail including its type, color, and any notable features. If you are not clear what exactly the object is, try to describe it as specifically as possible.",
    ]
    
    # Add evaluation questions for each metric
    for i, metric in enumerate(metrics):
        if include_descriptions:
            question = f"Is there any issue with the {metric}? \n{generate_ar_metrics_description(i)}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."
        else:
            question = f"Is there any issue with the {metric}? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."
        conversation.append(question)
    
    # Add final question about other issues
    conversation.append("Except Object Placement, Object Movement, Occlusion, Lighting, Artifacts and Rendering and Black Screen, have you found any other issues about Augmented Reality in this video?")
    
    return conversation

def generate_few_shot_prompt(examples):
    """Generate few-shot prompt with example images.
    
    Args:
        examples: List of example objects with ground_truth attribute
        
    Returns:
        String containing the few-shot prompt
    """
    prompt = "Here are some example AR screenshots with their quality assessment:\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Screenshot: <image>\n"
        prompt += f"Description:\n{example.ground_truth}\n\n"
    
    prompt += "Now, please understand the context of your task, and fully understand examples provided. Next, I will ask questions and please evaluate the AR effect in the Test Video.\n"
    return prompt

# Constants for easy access
METRICS = [
    "Object Placement",
    "Object Movement",
    "Occlusion",
    "Lighting", 
    "Visual Artifacts and Rendering Issues",
    "Black Screen"
]