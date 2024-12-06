async def generate_ai_summary(self, url: str, content: Union[str, Dict[str, Any]], content_type: str) -> Dict[str, Any]:
    try:
        # Get base LLM config
        llm_config_dict = self.config.llm_configs["content_analysis"].copy()
        
        # Remove non-LLMConfig fields
        system_prompt = llm_config_dict.pop('system_prompt', None)
        prompt_template = llm_config_dict.pop('prompt_template', None)
        llm_config = LLMConfig(**llm_config_dict)

        # Process content
        if isinstance(content, dict):
            content_text = content.get('text', '')[:self.config.content_max_length]
        else:
            content_text = str(content)[:self.config.content_max_length]

        # Create structured analysis request
        analysis_request = {
            "url": url,
            "content_type": content_type,
            "request": {
                "summary": "Provide a concise summary of the main points",
                "key_points": "List the key points about memecoins and market trends",
                "market_impact": "Analyze potential market impact and trends",
                "trading_implications": "Provide specific trading insights and recommendations"
            }
        }

        # Format the prompt
        formatted_prompt = f"""
        Analyze this content and provide insights in JSON format:

        URL: {url}
        CONTENT TYPE: {content_type}
        
        CONTENT:
        {content_text}

        Please provide analysis in the following JSON structure:
        {{
            "summary": "Brief overview of main points",
            "key_points": ["point 1", "point 2", ...],
            "market_impact": "Analysis of market implications",
            "trading_implications": "Specific trading insights"
        }}
        """

        # Create prompt context
        context_id = str(uuid.uuid4())
        context = LLMPromptContext(
            id=context_id,
            system_string="You are an expert financial analyst. Provide analysis in valid JSON format only.",
            new_message=formatted_prompt,
            llm_config=llm_config.dict(),
            use_history=False,
            source_id=context_id
        )

        # Process response with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                responses = await self.ai_utils.run_parallel_ai_completion([context])
                
                if responses and len(responses) > 0:
                    response = responses[0]
                    
                    # Try multiple approaches to extract valid JSON
                    if response.json_object:
                        return response.json_object.object
                    
                    if response.str_content:
                        # Clean the response string
                        content = response.str_content.strip()
                        
                        # Remove common markdown and code block markers
                        content = re.sub(r'^```json\s*', '', content)
                        content = re.sub(r'^```\s*', '', content)
                        content = re.sub(r'\s*```$', '', content)
                        
                        # Try parsing the cleaned content
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            # Try to extract JSON from within the text
                            json_match = re.search(r'({[\s\S]*})', content)
                            if json_match:
                                try:
                                    return json.loads(json_match.group(1))
                                except json.JSONDecodeError:
                                    pass
                            
                            # If JSON parsing fails, create structured response
                            return {
                                "summary": content[:500],  # First 500 chars as summary
                                "key_points": [line.strip() for line in content.split('\n') if line.strip()],
                                "market_impact": "Analysis pending",
                                "trading_implications": "Analysis pending"
                            }

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

        # Fallback response if all attempts fail
        return {
            "summary": "Content analysis failed after multiple attempts",
            "key_points": ["Analysis not available"],
            "market_impact": "Analysis not available",
            "trading_implications": "Analysis not available",
            "error": "Failed to generate valid JSON response",
            "url": url,
            "content_type": content_type
        }

    except Exception as e:
        logger.error(f"Error in AI summary generation: {str(e)}")
        return {
            "summary": "Error occurred during analysis",
            "key_points": ["Analysis failed"],
            "market_impact": "Analysis not available",
            "trading_implications": "Analysis not available",
            "error": str(e),
            "url": url,
            "content_type": content_type
        }